from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pdb

from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

def get_model(NB_GENRES = 9,
                noise_len= 100,
                len_seq= 32,
                nb_notes=9,
                kernal ="LSTM"
                ):
    return (discriminator(NB_GENRES,len_seq,nb_notes,kernal),
            generator(NB_GENRES,noise_len,len_seq,nb_notes,kernal))

def get_loss(cuda = True):
    print(cuda)
    return(conditioned_gan_loss(cuda = True))


class conditioned_gan_loss(torch.nn.Module):
    def __init__(self,cuda = True):
        super(conditioned_gan_loss, self).__init__()
        self.cuda = cuda
        self.f_loss = torch.nn.BCELoss()
        if self.cuda == True:
            self.weight = torch.tensor([1,1]).cuda()
            self.f_loss.cuda()
        else:
            self.weight = torch.tensor([1,1]).cpu()
            self.f_loss.cpu()

    def load_weight(self,weight):
        if self.cuda == True:
            self.weight = torch.tensor(weight).cuda()
            self.f_loss = torch.nn.BCELoss(weight = self.weight)
            self.f_loss.cuda()

        else:
            self.weight = torch.tensor(weight).cpu()
            self.f_loss = torch.nn.BCELoss(weight = self.weight)
            self.f_loss.cpu()


    def forward(self, pred, target):
        loss = self.f_loss(pred, target)
        return loss


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, input_channels = 128, nhead = 4, nhid = 2048, nlayers = 3, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'

        self.pos_encoder = PositionalEncoding(input_channels, dropout)
        encoder_layers = TransformerEncoderLayer(input_channels, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.input_channels = input_channels

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        #sz should be length of sequence
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask = None):
        #src should be [sequence length, batch size, embed dim]
        src = src * math.sqrt(self.input_channels)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        # output = self.decoder(output)
        return output

class discriminator(nn.Module):
    def __init__(self,
                NB_GENRES = 9,
                len_seq= 32,
                nb_notes=9,
                kernal = 'LSTM'):
        super(discriminator, self).__init__()
        self.NB_GENRES = NB_GENRES
        self.len_seq = len_seq
        self.nb_notes = nb_notes
        self.kernal = kernal

        self.embedding_block = nn.Sequential(
                                        torch.nn.Embedding(NB_GENRES,np.prod((len_seq, nb_notes))))

        if(kernal == "Transformer"):
            self.kernal_block = TransformerModel(input_channels = 18,nhead = 6,nlayers = 2,dropout=0.3)
            self.kernalout = torch.nn.Linear(18, 128)
        elif(kernal == "LSTM"):
            self.kernal_block = torch.nn.LSTM(18,hidden_size = 64,num_layers = 2, batch_first = False,bidirectional = True)

        self.MLP_block = nn.Sequential(
                                    torch.nn.Dropout(0.3),
                                    torch.nn.Linear(128, 512),
                                    torch.nn.LeakyReLU(0.01),
                                    torch.nn.Dropout(0.3),
                                    torch.nn.Linear(512, 256),
                                    torch.nn.LeakyReLU(0.01),
                                    torch.nn.Linear(256, 1))
        
        self.sg = torch.nn.Sigmoid()


    def forward(self, x,label):

        B = x.shape[0]
        label = self.embedding_block(label)#(B,1)->(B,1,288)
        label = label.squeeze(1) #(B,1,288)->(B,288)
        label = label.view(B,self.len_seq, self.nb_notes)#(B,288)->(B,32,9)
        x = torch.cat([x,label],axis = 2)#(B,32,9),(B,32,9)->(B,32,18)

        #Kernal block
        #=========================================
        x = x.permute(1,0,2)#(B,32,18)->(32,B,18)
        
        if(self.kernal == "Transformer"):
            x = self.kernal_block(x)
            x = self.kernalout(x)
        elif(self.kernal == "LSTM"):
            x,_ = self.kernal_block(x)
        
        x = x.permute(1,0,2)#(32,B,128)->(B,32,128)
        x = x[:,-1,:].view(B,-1)#Take the last output: (B,32,128)->(B,1,128)->(B,128)
        #=========================================

        x = self.MLP_block(x)#(B,128) -> (B,1)
        x = self.sg(x)
        return x


class generator(nn.Module):
    def __init__(self,NB_GENRES = 9,
                 noise_len= 100,
                 len_seq= 32,
                 nb_notes=9,
                 kernal = 'LSTM'):
        
        super(generator, self).__init__()
        self.NB_GENRES = NB_GENRES
        self.noise_len = noise_len
        self.len_seq = len_seq
        self.nb_notes = nb_notes
        self.kernal = kernal

        self.embedding_block = nn.Sequential(
                                            torch.nn.Embedding(NB_GENRES,noise_len))

        self.MLP_block = nn.Sequential(        
                                    torch.nn.Linear(noise_len, 512),
                                    torch.nn.LeakyReLU(0.2),
                                    torch.nn.BatchNorm1d(512, momentum=0.9),
                                    torch.nn.Linear(512, 1024),
                                    torch.nn.LeakyReLU(0.2),
                                    torch.nn.BatchNorm1d(1024, momentum=0.9)
                                    )

        self.drop1 = torch.nn.Dropout(0.3)

        if(kernal == "Transformer"):
            self.kernal_block = TransformerModel(input_channels = 32,nhead = 8,nlayers = 3,dropout=0.3)
            self.kernalout = torch.nn.Linear(32, nb_notes)
        elif(kernal == "LSTM"):
            self.kernal_block = torch.nn.LSTM(32,hidden_size = 128,num_layers = 3,batch_first = False,bidirectional = False)
            self.kernalout = torch.nn.Linear(128, nb_notes)

        self.sg = torch.nn.Sigmoid() 


    def forward(self, x,label):

        B = x.shape[0]
        label = self.embedding_block(label)#(B,1)->(B,1,100)
        label = label.squeeze(1)#(B,1,100)->(B,100)
        x = torch.mul(x,label)#(B,100),(B,100)->(B,100)

        x = self.MLP_block(x)#(B,100)->(B,1024)
        x = x.view(B,32,32)#(B,1024) ->(B,32,32)
        x = self.drop1(x) 

        #Kernal block
        #=========================================
        x = x.permute(1,0,2)#(B,32,32)->(32,B,32)


        if(self.kernal == "Transformer"):
            x = self.kernal_block(x)
            x = self.kernalout(x)
        elif(self.kernal == "LSTM"):
            x,_ = self.kernal_block(x)
            x = self.kernalout(x)

        x = x.permute(1,0,2)#(32,B,128)->(B,32,128)
        #=========================================

        x = self.sg(x) 
        return x