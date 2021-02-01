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


def get_model(NB_GENRES = 9,len_input= 100,len_seq= 32,nb_notes=9,sequence_model ="Transformer"):
    return (generator(NB_GENRES,len_input,len_seq,nb_notes,sequence_model),
            discriminator(NB_GENRES,len_input,len_seq,nb_notes,sequence_model))

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



def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
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


class discriminator(nn.Module):
    def __init__(self,NB_GENRES = 9,len_input= 100,len_seq= 32,nb_notes=9,sequence_model ="Transformer"):
        super(discriminator, self).__init__()
        self.NB_GENRES = NB_GENRES
        self.len_input = len_input
        self.len_seq = len_seq
        self.nb_notes = nb_notes
        self.sequence_model = sequence_model


        self.embedding1 = torch.nn.Embedding(NB_GENRES,np.prod((len_seq, nb_notes)))
        

        #LSTM block
        #=========================================
        if(self.sequence_model  =="LSTM"):

            self.bidirectional1 = torch.nn.LSTM(18,hidden_size = 64,batch_first = True,bidirectional = True)
            self.bidirectional2 = torch.nn.LSTM(128,hidden_size = 64,batch_first = True,bidirectional = True)
            
            self.drop1 = torch.nn.Dropout(0.3)
            self.l1 = torch.nn.Linear(128, 512)
            self.LeakyReLU1 = torch.nn.LeakyReLU(0.01)
            
            self.drop2 = torch.nn.Dropout(0.3)
            self.l2 = torch.nn.Linear(512, 256)
            self.LeakyReLU2 = torch.nn.LeakyReLU(0.01)
            self.l3 = torch.nn.Linear(256, 1)
        #=========================================

        
        
        # #transfermer block
        #=========================================
        if(self.sequence_model  =="Transformer"):
            self.src_mask = None
            self.pos_encoder = PositionalEncoding(18, dropout = 0.1)
            self.encoder_layers = TransformerEncoderLayer(18, nhead = 2 , dim_feedforward = 2048, dropout = 0.1)
            self.transformer_encoder = TransformerEncoder(self.encoder_layers, num_layers=2)
            self.decoder = torch.nn.Linear(18, 1)
            self.decoder2 = torch.nn.Linear(32, 1)
        #=========================================

        self.sg = torch.nn.Sigmoid()


    def forward(self, x,label,has_mask = True):

        B = x.shape[0]
        label = self.embedding1(label)
        label = label.view(-1,self.len_seq, self.nb_notes)
        x = torch.cat([x,label],axis = 2)




        #LSTM block
        # #=========================================
        if(self.sequence_model  =="LSTM"):
            x,_= self.bidirectional1(x)
            x,_ = self.bidirectional2(x)
            x = x[:,-1,:].view(B,-1)
            x = self.drop1(x)
            x = self.l1(x)
            x = self.LeakyReLU1(x)

            x = self.drop2(x)
            x = self.l2(x)
            x = self.LeakyReLU2(x)
            x = self.l3(x)
        #=========================================


                
        # #transfermer block
        #=========================================
        if(self.sequence_model  =="Transformer"):
            if has_mask:
                device = x.device
                if self.src_mask is None or self.src_mask.size(0) != len(x):
                    mask = generate_square_subsequent_mask(len(x)).to(device)
                    self.src_mask = mask
            else:
                self.src_mask = None


            x = self.pos_encoder(x)
            x = self.transformer_encoder(x, self.src_mask)
            x = self.decoder(x)
            x = self.decoder2(x.view(B,-1))
        #=========================================


        x = self.sg(x) 
        return x


class generator(nn.Module):
    def __init__(self,NB_GENRES = 9,len_input= 100,len_seq= 32,nb_notes=9,sequence_model ="Transformer"):
        super(generator, self).__init__()
        self.NB_GENRES = NB_GENRES
        self.len_input = len_input
        self.len_seq = len_seq
        self.nb_notes = nb_notes
        self.sequence_model = sequence_model


        self.embedding1 = torch.nn.Embedding(NB_GENRES,len_input)        
        self.l1 = torch.nn.Linear(len_input, 512)
        self.LeakyReLU1 = torch.nn.LeakyReLU(0.2)
        self.bn1 = torch.nn.BatchNorm1d(512, momentum=0.9)

        self.l2 = torch.nn.Linear(512, 1024)
        self.LeakyReLU2 = torch.nn.LeakyReLU(0.2)
        self.bn2 = torch.nn.BatchNorm1d(1024, momentum=0.9)

        self.drop1 = torch.nn.Dropout(0.3)



        #LSTM block
        # #=========================================
        if(self.sequence_model  =="LSTM"):

            self.LSTM1 = torch.nn.LSTM(32,hidden_size = 128,batch_first = True,bidirectional = False)
            self.LSTM2 = torch.nn.LSTM(128,hidden_size = 128,batch_first = True,bidirectional = False)
            self.LSTM3 = torch.nn.LSTM(128,hidden_size = 9,batch_first = True,bidirectional = False)
        
        # #=========================================

        
        
        # #transfermer block
        # #=========================================
        if(self.sequence_model  =="Transformer"):

            self.src_mask = None
            self.pos_encoder = PositionalEncoding(32, dropout = 0.1)
            self.encoder_layers = TransformerEncoderLayer(32, nhead = 2 , dim_feedforward = 2048, dropout = 0.1)
            self.transformer_encoder = TransformerEncoder(self.encoder_layers, num_layers=3)
            self.decoder = torch.nn.Linear(32, 9)
        
        # #=========================================


        self.sg = torch.nn.Sigmoid() 


    def forward(self, x,label,has_mask=True):

        B = x.shape[0]
        label = self.embedding1(label)

        label = label.view(B,self.len_input)

        x = torch.mul(x,label)

        x = self.l1(x) 
        x = self.LeakyReLU1(x) 
        x = self.bn1(x) 

        x =self.l2(x)  
        x =self.LeakyReLU2(x) 
        x =self.bn2(x)  
        x = x.view(B,32,32)
        x = self.drop1(x) 



        #LSTM block
        # #=========================================
        if(self.sequence_model =="LSTM"):
            x,_ =self.LSTM1(x)  
            x,_ =self.LSTM2(x)  
            x,_ =self.LSTM3(x) 
        # #=========================================




        # #transfermer block
        # #=========================================
        if(self.sequence_model  =="Transformer"):
            if has_mask:
                device = x.device
                if self.src_mask is None or self.src_mask.size(0) != len(x):
                    mask = generate_square_subsequent_mask(len(x)).to(device)
                    self.src_mask = mask
            else:
                self.src_mask = None
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x, self.src_mask)
            x = self.decoder(x)
        # #=========================================





        x = self.sg(x) 



        return x