"""
Utility functions for the model
"""

# importing libraries
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import softmax


def subsequent_mask(size):
    """
    Function to compute the mask used in attention layer of decoder

    INPUT:
    size - (int) horizon size

    OUTPUT:
    mask - (torch tensor) boolean array to mask out the data in decoder
    """

    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    mask = torch.from_numpy(mask) == 0

    return mask


def attention(Q, K, V, mask=None, dropout=None):
    """
    Function to compute the attention from given Q, K and V values 

    INPUT:
    Q - (torch tensor) query for the transformer. Shape = (B, H, N, C)
    K - (torch tensor) keys for the transformer. Shape = (B, H, N, C)
    V - (torch tensor) values for the transformer. Shape = (B, H, N, C) 
    mask - (torch tensor) mask for decoder multi head attention layer
    dropout - (float) dropout percentage

    OUTPUT:
    attn_output - (torch tensor) output of the multi head attention layer. Shape = (B, H, N, C)
    """

    # finding the embedding size
    new_emb_size = Q.shape[0]
    # calculating attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(new_emb_size)

    # applying mask on the attention
    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)

    # applying softmax layer and calculating prob of attention
    p_attn = softmax(scores, dim=-1)

    # applying dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    # multiplying the prob of attentiom with Values (V)  
    attn_output = torch.matmul(p_attn, V)

    return attn_output

def cosine_scheduler(t, eta_max, T):
    """
    Function to implement cosine scheduler
    """

    T_0 = T/5

    if t <= T_0:
        lr = 1e-6 + ((t/T_0) * eta_max)
    else:
        lr = 1e-8 + (eta_max * np.cos((np.pi/2)*((t-T_0)/(T-T_0))))
    
    return lr


def learning_rate_finder(tf_model, optimizer, train_loader, iterations, device, mean, std, increment=1.1):
    """
    Function to perform the "learning rate finder" algorithm.
    """

    # initilizing array to store training loss of each minibatch
    train_loss = []
    # initializing the array to store the learning rates
    learning_rates = []

    for idx, data in enumerate(train_loader):
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
        dec_target_mask = subsequent_mask(dec_input.shape[1]).repeat(dec_input.shape[0],1,1).to(device)

        # forward pass 
        optimizer.zero_grad()
        predictions = tf_model.forward(enc_input, dec_input, dec_source_mask, dec_target_mask)

        # calculating loss using pairwise distance of all predictions
        loss = F.pairwise_distance(predictions[:, :,0:2].contiguous().view(-1, 2),
                                    ((data['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).\
                                    contiguous().view(-1, 2).to(device)).mean() + \
                                    torch.mean(torch.abs(predictions[:,:,2]))
        train_loss.append(loss.item())             
        # changing the learning rate 
        for param in optimizer.param_groups:
            learning_rates.append(param['lr'])
            param['lr'] *= increment
        
        # updating weights
        loss.backward()
        optimizer.step()

        if idx == iterations:
            break

    return train_loss, learning_rates