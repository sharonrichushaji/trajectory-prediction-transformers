"""
Script to create a torch Dataset for the TrajNet Dataset. Script adapted from https://arxiv.org/abs/2003.08111
"""

# importing libraries
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch
import random
import scipy.spatial
import scipy.io


class TFDataset(Dataset):
    """
    Class to create the torch Dataset object
    """

    def __init__(self,data,name,mean,std):
        super(TFDataset,self).__init__()

        self.data=data  # dictionary
        self.name=name  # string

        self.mean= mean # numpy (4,)
        self.std = std  # numpy (4,)

    def __len__(self):
        return self.data['src'].shape[0]


    def __getitem__(self,index):
        return {'src':torch.Tensor(self.data['src'][index]),
                'trg':torch.Tensor(self.data['trg'][index]),
                'frames':self.data['frames'][index],
                'seq_start':self.data['seq_start'][index],
                'dataset':self.data['dataset'][index],
                'peds': self.data['peds'][index],
                }


def create_dataset(dataset_folder, dataset_name, val_size, gt, horizon, delim="\t", train=True, eval=False, verbose=False):
    """
    Function to import .txt file and create the torch Dataset.
    The function splits the dataset into train, val and test.

    INPUT:
    dataset_folder - (str) path to the folder containing the dataset
    dataset_name - (str) name of the dataset
    val_size - (int) number of datapoints in validation set
    gt - (int) size of the input observations to be passed to the encoder
    horizon - (int) size of the input observations to be passed to the decoder
    delim - (str) delimeter needed to read the dataset csv file
    train - (bool) boolean indicating if training dataset needs to be created. Default value = True
    eval - (bool) boolean indicating if testing dataset needs to be created. Default value = False
    verbose - (bool) boolean indicating if printing in required during code execution. Default value = False

    OUTPUT:

    object of the TFDataset class
    """

    # finding the path to the required datasets
    if train==True:
        datasets_list = os.listdir(os.path.join(dataset_folder,dataset_name, "train"))
        full_dt_folder=os.path.join(dataset_folder,dataset_name, "train")
    if train==False and eval==False:
        datasets_list = os.listdir(os.path.join(dataset_folder, dataset_name, "val"))
        full_dt_folder = os.path.join(dataset_folder, dataset_name, "val")
    if train==False and eval==True:
        datasets_list = os.listdir(os.path.join(dataset_folder, dataset_name, "test"))
        full_dt_folder = os.path.join(dataset_folder, dataset_name, "test")


    datasets_list=datasets_list
    data={}
    data_src=[]
    data_trg=[]
    data_seq_start=[]
    data_frames=[]
    data_dt=[]
    data_peds=[]

    val_src = []
    val_trg = []
    val_seq_start = []
    val_frames = []
    val_dt = []
    val_peds=[]

    if verbose:
        print("start loading dataset")
        print("validation set size -> %i"%(val_size))


    # enumerating through all the .txt files in the required dataset folder
    for i_dt, dt in enumerate(datasets_list):
        if verbose:
            print("%03i / %03i - loading %s"%(i_dt+1,len(datasets_list),dt))
        raw_data = pd.read_csv(os.path.join(full_dt_folder, dt), delimiter=delim,
                                        names=["frame", "ped", "x", "y"],usecols=[0,1,2,3],na_values="?")

        raw_data.sort_values(by=['frame','ped'], inplace=True)

        inp,out,info=get_strided_data_clust(raw_data,gt,horizon,1)

        dt_frames=info['frames']
        dt_seq_start=info['seq_start']
        dt_dataset=np.array([i_dt]).repeat(inp.shape[0])
        dt_peds=info['peds']


        # creating validation data only if the training data is > 2.5 * validation data size
        if val_size>0 and inp.shape[0]>val_size*2.5:
            if verbose:
                print("created validation from %s" % (dt))
            k = random.sample(np.arange(inp.shape[0]).tolist(), val_size)
            val_src.append(inp[k, :, :])
            val_trg.append(out[k, :, :])
            val_seq_start.append(dt_seq_start[k, :, :])
            val_frames.append(dt_frames[k, :])
            val_dt.append(dt_dataset[k])
            val_peds.append(dt_peds[k])
            inp = np.delete(inp, k, 0)
            out = np.delete(out, k, 0)
            dt_frames = np.delete(dt_frames, k, 0)
            dt_seq_start = np.delete(dt_seq_start, k, 0)
            dt_dataset = np.delete(dt_dataset, k, 0)
            dt_peds = np.delete(dt_peds,k,0)
        elif val_size>0:
            if verbose:
                print("could not create validation from %s, size -> %i" % (dt,inp.shape[0]))

        data_src.append(inp)                # (num_datapoints in current dataset file, gt_size, 4)
        data_trg.append(out)                # (num_datapoints in current dataset file, horizon, 4)
        data_seq_start.append(dt_seq_start) # (num_datapoints in current dataset file, 1, 2)
        data_frames.append(dt_frames)       # (num_datapoints in current dataset file, 20)
        data_dt.append(dt_dataset)          # (num_datapoints in current dataset file,) data set ids
        data_peds.append(dt_peds)           # (num_datapoints in current dataset file,) pedestrian ids



    # concatenating all the dataset files in the current dataset folder
    data['src'] = np.concatenate(data_src, 0)
    data['trg'] = np.concatenate(data_trg, 0)
    data['seq_start'] = np.concatenate(data_seq_start, 0)
    data['frames'] = np.concatenate(data_frames, 0)
    data['dataset'] = np.concatenate(data_dt, 0)
    data['peds'] = np.concatenate(data_peds, 0)
    data['dataset_name'] = datasets_list

    # finding the mean and standard deviation of x and y positions of all datapoints
    mean= data['src'].mean((0,1))
    std= data['src'].std((0,1))

    # concatenate all validation dataset files
    if val_size>0:
        data_val={}
        data_val['src']=np.concatenate(val_src,0)
        data_val['trg'] = np.concatenate(val_trg, 0)
        data_val['seq_start'] = np.concatenate(val_seq_start, 0)
        data_val['frames'] = np.concatenate(val_frames, 0)
        data_val['dataset'] = np.concatenate(val_dt, 0)
        data_val['peds'] = np.concatenate(val_peds, 0)

        return TFDataset(data, "train", mean, std), TFDataset(data_val, "validation", mean, std)

    return TFDataset(data, "train", mean, std), None


def get_strided_data_clust(dt, gt_size, horizon, step):
    """
    splitting the data into clusters of a size (gt_size + horizon)

    INPUT:
    dt - (pandas df) dataframe for the dataset
    gt_size - (int) size of the input observations to be passed to the encoder
    horizon - (int) size of the input observations to be passed to the decoder 
    step - (int) integer indicating the stride/window shift

    OUTPUT:
    inp - (numpy arrays) input data to encoder. Shape = (num_datapoints, gt_size, 4)
    out - (numpy arrays) input data to decoder. Shape = (num_datapoints, horizon, 4)
    info - (dictionary) meta for dataset
    """

    inp_te = []
    dtt = dt.astype(np.float32)
    raw_data = dtt

    ped = raw_data.ped.unique()
    frame=[]
    ped_ids=[]

    # clustering the dataset by looping through all unique pedestrian ID's
    for p in ped:
        for i in range(1+(raw_data[raw_data.ped == p].shape[0] - gt_size - horizon) // step):
            frame.append(dt[dt.ped == p].iloc[i * step:i * step + gt_size + horizon, [0]].values.squeeze())
            # print("%i,%i,%i" % (i * 4, i * 4 + gt_size, i * 4 + gt_size + horizon))
            inp_te.append(raw_data[raw_data.ped == p].iloc[i * step:i * step + gt_size + horizon, 2:4].values)
            ped_ids.append(p)

    frames=np.stack(frame)
    inp_te_np = np.stack(inp_te)
    ped_ids=np.stack(ped_ids)

    # calculating the speed of the pedestrian between 2 frames
    inp_speed = np.concatenate((np.zeros((inp_te_np.shape[0],1,2)),inp_te_np[:,1:,0:2] - inp_te_np[:, :-1, 0:2]),1)

    # concatenating position and speed information
    inp_norm=np.concatenate((inp_te_np,inp_speed),2)

    inp_mean=np.zeros(4)
    inp_std=np.ones(4)

    inp = inp_norm[:,:gt_size]
    out = inp_norm[:,gt_size:]

    # seq_start - starting x,y position in every window
    info = {'mean': inp_mean, 'std': inp_std, 'seq_start': inp_te_np[:, 0:1, :].copy(),'frames':frames,'peds':ped_ids}

    return inp, out, info


def distance_metrics(target, preds):
    """
    Function to calculate MAD (mean average displacement) and FAD (final average displacement)

    INPUT:
    target - (numpy array) ground truth values for pedestrian positions
    preds - (numpy array) predicted values for pedestrian positions

    OUTPUT:
    mad - (numpy array) MAD value
    fad - (numpy array) FAD value
    errors - (numpy array) error value
    """
    
    errors = np.zeros(preds.shape[:-1])
    for i in range(errors.shape[0]):
        for j in range(errors.shape[1]):
            errors[i, j] = scipy.spatial.distance.euclidean(target[i, j], preds[i, j])

    mad = errors.mean()
    fad = errors[:,-1].mean()

    return mad, fad, errors