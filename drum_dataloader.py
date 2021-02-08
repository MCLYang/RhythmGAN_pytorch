import os
import os.path as osp
import numpy as np
import torch

class drum_dataloader():
    def __init__(self,
                 root,
                 ):
        print("Loading post_preccessed data....")
        self.matrices_onsets = np.load(root)['onsets']
        self.matrices_velos = np.load(root)['velocities']
        self.matrices_offsets = np.load(root)['offsets']
        self.matrices_genres = np.load(root)['genre_ids'].reshape(-1,1)
        self.GENRES = np.load(root)['genres']
        print("Loading successful")
        print("X shape: ",self.matrices_velos.shape)
        print("Label shape: ",self.matrices_genres.shape)

    def __getitem__(self, index):
        drum = self.matrices_velos[index, :, :]
        label = self.matrices_genres[index,:]

        drum = torch.FloatTensor(drum)
        label = torch.LongTensor(label)

        return(drum,label) 

    def __len__(self):
        return len(self.matrices_velos)