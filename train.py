"""
Script to perform model training
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm 
import numpy as np
import os
import dataloader 
import model
import utils
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # defining model save location
    save_location = "./models"
    # defining dataset locations
    dataset_folder = "./datasets"
    dataset_name = "raw"
    # setting validation size. if val_size = 0, split percentage is 80-20
    val_size = 0
    # length of sequence given to encoder
    gt = 8
    # length of sequence given to decoder
    horizon = 12

    # creating torch datasets
    train_dataset, _ = dataloader.create_dataset(dataset_folder, dataset_name, val_size, \
        gt, horizon, delim="\t", train=True)
    val_dataset, _ = dataloader.create_dataset(dataset_folder, dataset_name, val_size, \
        gt, horizon, delim="\t", train=False)
    test_dataset, _ = dataloader.create_dataset(dataset_folder, dataset_name, val_size, \
        gt, horizon, delim="\t", train=False, eval=True)

    # defining batch size
    batch_size = 64

    # creating torch dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0)
    
    # calculating the mean and standard deviation of velocities of the entire dataset
    mean=torch.cat((train_dataset[:]['src'][:,1:,2:4],train_dataset[:]['trg'][:,:,2:4]),1).mean((0,1))
    std=torch.cat((train_dataset[:]['src'][:,1:,2:4],train_dataset[:]['trg'][:,:,2:4]),1).std((0,1))
    means=[]
    stds=[]
    for i in np.unique(train_dataset[:]['dataset']):
        ind=train_dataset[:]['dataset']==i
        means.append(torch.cat((train_dataset[:]['src'][ind, 1:, 2:4], train_dataset[:]['trg'][ind, :, 2:4]), 1).mean((0, 1)))
        stds.append(
            torch.cat((train_dataset[:]['src'][ind, 1:, 2:4], train_dataset[:]['trg'][ind, :, 2:4]), 1).std((0, 1)))
    mean=torch.stack(means).mean(0)
    std=torch.stack(stds).mean(0)

    # performing training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # creating model
    encoder_ip_size = 2
    decoder_ip_size = 3
    model_op_size = 3
    emb_size = 512
    num_heads = 8
    ff_hidden_size = 2048
    n = 6
    dropout=0.1

    tf_model = model.TFModel(encoder_ip_size, decoder_ip_size, model_op_size, emb_size, \
                    num_heads, ff_hidden_size, n, dropout=0.1).to(device)
    
    # number of iterations for LRF
    iterations = 70

    # creating optimizer
    optimizer = torch.optim.SGD(tf_model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-3, nesterov=True)
    # optimizer = torch.optim.Adam(tf_model.parameters(), lr=1e-4)

    train_loss, learning_rates = utils.learning_rate_finder(tf_model, optimizer, train_loader, iterations, device, mean, std)
    eta_star = learning_rates[np.argmin(np.array(train_loss))]
    eta_max = eta_star/10
    print("Value of eta max is: {:.4f}".format(eta_max))

    # plotting results
    plt.figure()
    plt.plot(learning_rates, train_loss)
    plt.xlabel("Learning rates")
    plt.ylabel("Training loss")
    plt.xscale('log')
    plt.title("Learning Rate Finder Algorithm")
    plt.show()
    
    # number of epochs 
    epochs = 100

    # metric variables
    training_loss = []
    validation_loss = []
    val_mad = []
    val_fad = []

    # finding the total number of weight updates for the network
    T = epochs * len(train_loader)
    # initializing variable to track the number of weight updates
    weight_update = 0
    # initializing variable to store the changing learning rate
    learning_rate = []

    for epoch in tqdm(range(epochs)):
        # TRAINING MODE
        tf_model.train()
        
        # training batch variables
        train_batch_loss = 0

        for idx, data in enumerate(train_loader):
            # changing the learning rate based on cosine scheduler
            lr = utils.cosine_scheduler(weight_update, eta_max, T)
            for param in optimizer.param_groups:
                learning_rate.append(lr)
                param['lr'] = lr
            weight_update += 1

            # getting encoder input data
            enc_input = (data['src'][:,1:,2:4].to(device)-mean.to(device))/std.to(device)

            # getting decoder input data
            target = (data['trg'][:,:-1,2:4].to(device)-mean.to(device))/std.to(device)
            target_append = torch.zeros((target.shape[0],target.shape[1],1)).to(device)
            target = torch.cat((target,target_append),-1)
            start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0],1,1).to(device)
            dec_input = torch.cat((start_of_seq, target), 1)

            # getting masks for decoder
            dec_source_mask = torch.ones((enc_input.shape[0], 1,enc_input.shape[1])).to(device)
            dec_target_mask = utils.subsequent_mask(dec_input.shape[1]).repeat(dec_input.shape[0],1,1).to(device)

            # forward pass 
            optimizer.zero_grad()
            predictions = tf_model.forward(enc_input, dec_input, dec_source_mask, dec_target_mask)

            # calculating loss using pairwise distance of all predictions
            loss = F.pairwise_distance(predictions[:, :,0:2].contiguous().view(-1, 2),
                                       ((data['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).\
                                        contiguous().view(-1, 2).to(device)).mean() + \
                                        torch.mean(torch.abs(predictions[:,:,2]))
            train_batch_loss += loss.item()
            
            # updating weights
            loss.backward()
            optimizer.step()

        training_loss.append(train_batch_loss/len(train_loader))
        print("Epoch {}/{}....Training loss = {:.4f}".format(epoch+1, epochs, training_loss[-1]))
    

        # validation loop
        if (epoch+1)%5 == 0:
            with torch.no_grad():
                # EVALUATION MODE
                tf_model.eval()
                
                # validation variables
                batch_val_loss=0
                gt = []
                pr = []

                for id_b, data in enumerate(val_loader):
                    # storing groung truth 
                    gt.append(data['trg'][:, :, 0:2])

                    # input to encoder input
                    val_input = (data['src'][:,1:,2:4].to(device)-mean.to(device))/std.to(device)

                    # input to decoder
                    start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(val_input.shape[0], 1, 1).to(device)
                    dec_inp = start_of_seq
                    # decoder masks
                    dec_source_mask = torch.ones((val_input.shape[0], 1, val_input.shape[1])).to(device)
                    dec_target_mask = utils.subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(device)

                    # prediction till horizon lenght
                    for i in range(horizon):
                        # getting model prediction
                        model_output = tf_model.forward(val_input, dec_inp, dec_source_mask, dec_target_mask)
                        # appending the predicition to decoder input for next cycle
                        dec_inp = torch.cat((dec_inp, model_output[:, -1:, :]), 1)

                    # calculating loss using pairwise distance of all predictions
                    val_loss = F.pairwise_distance(dec_inp[:,1:,0:2].contiguous().view(-1, 2),
                                            ((data['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).\
                                                contiguous().view(-1, 2).to(device)).mean() + \
                                                torch.mean(torch.abs(dec_inp[:,1:,2]))
                    batch_val_loss += val_loss.item()

                    # calculating the position for each time step of prediction based on velocity
                    preds_tr_b = (dec_inp[:, 1:, 0:2]*std.to(device) + mean.to(device)).cpu().numpy().cumsum(1) + \
                        data['src'][:,-1:,0:2].cpu().numpy()

                    pr.append(preds_tr_b)
                validation_loss.append(batch_val_loss/len(val_loader))

                # calculating mad and fad evaluation metrics
                gt = np.concatenate(gt, 0)
                pr = np.concatenate(pr, 0)
                mad, fad, _ = dataloader.distance_metrics(gt, pr)
                val_mad.append(mad)
                val_fad.append(fad)

                print("Epoch {}/{}....Validation mad = {:.4f}, Validation fad = {:.4f}".format(epoch+1, epochs, mad, fad))

        # Saving model, loss and error log files
        torch.save({
            'model_state_dict': tf_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_loss': training_loss,
            'validation_loss': validation_loss,
            'val_mad': val_mad,
            'val_fad':val_fad,
            'learning_rate':learning_rate
            }, os.path.join(save_location, 'epoch{}.pth'.format(epoch+1)))
    

    # loading saved model file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded_file = torch.load(os.path.join(save_location, 'epoch150.pth'), map_location=torch.device(device))

    # creating model and loading weights
    encoder_ip_size = 2
    decoder_ip_size = 3
    model_op_size = 3
    emb_size = 512
    num_heads = 8
    ff_hidden_size = 2048
    n = 6
    dropout=0.1

    model_loaded = model.TFModel(encoder_ip_size, decoder_ip_size, model_op_size, emb_size, \
                    num_heads, ff_hidden_size, n, dropout=0.1)
    model_loaded = model_loaded.to(device)
    model_loaded.load_state_dict(loaded_file['model_state_dict'])

    # loading training metric variables
    training_loss = loaded_file['training_loss']
    validation_loss = loaded_file['validation_loss']
    val_mad = loaded_file['val_mad']
    val_fad = loaded_file['val_fad']
    learning_rate = loaded_file['learning_rate']

    # plotting training loss
    plt.figure()
    plt.plot(training_loss)
    plt.xlabel("Number of epochs")
    plt.ylabel("Training loss")
    plt.title("Training loss VS Number of Epochs")

    # plotting validation loss
    plt.figure()
    plt.plot(validation_loss)
    plt.xlabel("Number of epochs")
    plt.ylabel("Validation loss")
    plt.title("Validation loss VS Number of Epochs")

    # plotting training and validation loss together
    plt.figure()
    plt.plot(loaded_file['training_loss'], label="training loss")
    plt.plot(np.arange(1,100,5), loaded_file['validation_loss'], label="validation loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.title("Training v/s Validation loss")
    plt.savefig("loss.png")

    # plotting learning rate for model
    plt.figure()
    plt.plot(learning_rate)
    plt.xlabel("Number of epochs")
    plt.ylabel("learning_rate")
    plt.title("Learning_rate VS Number of Epochs")

    # plotting MAD
    plt.figure()
    plt.plot(np.arange(1,100,5), loaded_file['val_mad'], label="validation MAD")
    plt.xlabel("Epochs")
    plt.ylabel("MAD (m)")
    plt.title("Mean Average Displacement")
    plt.savefig("mad.png")

    # plotting FAD
    plt.figure()
    plt.plot(np.arange(1,100,5), loaded_file['val_fad'], label="validation FAD")
    plt.xlabel("Epochs")
    plt.ylabel("FAD (m)")
    plt.title("Final Average Displacement")
    plt.savefig("fad.png")

    plt.show()

    # Running the validation loop to generate prediction trajectories on validation data
    validation_loss = []
    val_mad = []
    val_fad = []

    with torch.no_grad():
        # EVALUATION MODE
        model_loaded.eval()
        
        # validation variables
        batch_val_loss=0
        gt = []
        pr = []
        obs = []

        for id_b, data in enumerate(val_loader):
            # storing groung truth 
            gt.append(data['trg'][:, :, 0:2])
            obs.append(data['src'][:,:, 0:2])
            # input to encoder input
            val_input = (data['src'][:,1:,2:4].to(device)-mean.to(device))/std.to(device)

            # input to decoder
            start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(val_input.shape[0], 1, 1).to(device)
            dec_inp = start_of_seq
            # decoder masks
            dec_source_mask = torch.ones((val_input.shape[0], 1, val_input.shape[1])).to(device)
            dec_target_mask = utils.subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(device)

            # prediction till horizon lenght
            for i in range(horizon):
                # getting model prediction
                model_output = model_loaded.forward(val_input, dec_inp, dec_source_mask, dec_target_mask)
                # appending the predicition to decoder input for next cycle
                dec_inp = torch.cat((dec_inp, model_output[:, -1:, :]), 1)

            # calculating loss using pairwise distance of all predictions
            val_loss = F.pairwise_distance(dec_inp[:,1:,0:2].contiguous().view(-1, 2),
                                    ((data['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).\
                                        contiguous().view(-1, 2).to(device)).mean() + \
                                        torch.mean(torch.abs(dec_inp[:,1:,2]))
            batch_val_loss += val_loss.item()

            # calculating the position for each time step of prediction based on velocity
            preds_tr_b = (dec_inp[:, 1:, 0:2]*std.to(device) + mean.to(device)).cpu().numpy().cumsum(1) + \
                data['src'][:,-1:,0:2].cpu().numpy()

            pr.append(preds_tr_b)
            validation_loss.append(batch_val_loss/len(val_loader))

        # calculating mad and fad evaluation metrics
        gt = np.concatenate(gt, 0)
        pr = np.concatenate(pr, 0)
        obs = np.concatenate(obs, 0)
        mad, fad, _ = dataloader.distance_metrics(gt, pr)
        val_mad.append(mad)
        val_fad.append(fad)

    # plotting the predicted and ground truth trajectories
    idx = np.random.randint(0, gt.shape[0])
    plt.figure()
    plt.scatter(gt[idx,:,0],gt[idx,:,1], color='green', label="Ground truth")
    plt.scatter(pr[idx,:,0],pr[idx,:,1], color='orange',label="Predictions")
    plt.scatter(obs[idx,:,0], obs[idx,:,1], color='b', label="Observations")
    plt.legend()
    plt.xlim(-8, 18)
    plt.ylim(-11, 15)
    plt.title("Trajectory Visualization in camera frame")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.savefig("traj_{}".format(idx))
    
    plt.show()