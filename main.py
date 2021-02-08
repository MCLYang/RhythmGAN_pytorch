import sys
from argparse import ArgumentParser
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # only relevant to my own environment


from tqdm import tqdm
import numpy as np
import pdb
from glob import glob
import pandas as pd
from pathlib import Path
import time
from collections import OrderedDict
import random
import pandas as pd
import importlib
import shutil

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import SubsetRandomSampler
from torch.autograd import Variable

from weight_init import weight_init
from drum_dataloader import drum_dataloader
from metrics_manager import metrics_manager
from rhythm_can.constants import *
from rhythm_can.utils import *

MAX_LOSS_RATIO = 3

def opt_global_inti():
        parser = ArgumentParser()
    #data
        # parser.add_argument('--dataset_root', type=str, default="./data/matrices_drum_gm_clean.npz", help="dataset path")
        parser.add_argument('--dataset_root', type=str, default="./data/matrices_drum_gm_clean_no_fill.npz", help="dataset path")
        parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=32)
        parser.add_argument('--noise_len', type=int, default = 100,help='length of noise vector')
        parser.add_argument('--shuffle', type=lambda x: (str(x).lower() == 'true'),default=True ,help="if shuffle the dataset")
        # parser.add_argument('--train_size', type=float,default=0.9 ,help="represent the proportion of the dataset to include in the train split")

    #model parameters
        parser.add_argument('--model', type=str,default='conditioned_gan' ,help="[conditioned_gan,..]")
        parser.add_argument('--kernal', type=str, default='Transformer', help='type of recurrent net (LSTM, Transformer)')

    #optimizer parameters
        parser.add_argument('--d_init_lr', type=float,default= 0.0001,help="discriminator optimizer initial learning rate")
        # parser.add_argument('--d_step_size', type=int,default=300 ,help="how many epochs to update the lr")#for LSTM
        parser.add_argument('--d_step_size', type=int,default=100 ,help="how many epochs to update the lr")#for Transformer
        parser.add_argument('--d_gamma', type=float,default=0.5 ,help="decay_rate")
        parser.add_argument('--g_init_lr', type=str,default=0.001,help="generator optimizer initial learning rate")
        # parser.add_argument('--g_step_size', type=int,default=300,help="how many epochs to update the lr"))#for LSTM
        parser.add_argument('--g_step_size', type=int,default=100,help="how many epochs to update the lr")#for Transformer
        parser.add_argument('--g_gamma', type=float,default=0.5,help="decay_rate")
        # parser.add_argument('--synchonization', type=str,default='BN' ,help="[BN,BN_syn,Instance]")

    #training parameters
        parser.add_argument('--apex', type=lambda x: (str(x).lower() == 'true'),default=False ,help="apexFF16")#Not implement yet
        parser.add_argument('--cuda', type=lambda x: (str(x).lower() == 'true'),default=True ,help="Using GPU or Not")
        parser.add_argument('--num_gpu', type=int,default=1,help="num_gpu")
        parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
        parser.add_argument('--epoch_max', type=int,default=1001,help="epoch_max")
        parser.add_argument('--K_unrolled_d', type=int,default=5,help="train Discriminator (K_unrolled) times per epoch")
        parser.add_argument('--K_unrolled_g', type=int,default=1,help="train Generator (K_unrolled) times per epoch")
        parser.add_argument('--debug', type=lambda x: (str(x).lower() == 'true'),default=False,help="is task for debugging?False for load entire dataset")
        parser.add_argument('--save_perEpoch', type=int,default=200,help="save_perEpoch")
        parser.add_argument('--vis_perEpoch', type=int,default=10,help="vis_perEpoch")
        parser.add_argument('--save_fig', type=lambda x: (str(x).lower() == 'true'),default=True,help="Save the sample plot during the testing")
        parser.add_argument('--show_fig', type=lambda x: (str(x).lower() == 'true'),default=False,help="show the sample plot during the testing")


    #wandb config(optional)
        parser.add_argument('--wandb', type=lambda x: (str(x).lower() == 'true'),default=False ,help="Use wandb or not")
        parser.add_argument('--wandb_history', type=lambda x: (str(x).lower() == 'true'),default=False ,help="load wandb history")
        parser.add_argument('--wandb_id', type=str,default='',help="")
        parser.add_argument('--wandb_file', type=str,default='',help="")
        parser.add_argument('--unsave_epoch', type=int,default=0,help="")
        parser.add_argument('--load_pretrain', type=str,default='',help="root load_pretrain")
        parser.add_argument('--wd_project', type=str,default="Creative_GAN",help="")
        args = parser.parse_args()
        return args


def save_model(package,root):
    torch.save(package,root)

def setSeed(seed = 2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def convert_state_dict(state_dict):
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def creating_new_model(opt):
    print('----------------------Creating model----------------------')
    opt.time = time.ctime()
    opt.epoch_ckpt = 0
    opt.d_loss = 0
    opt.g_loss = 0

    module_name = 'model.'+opt.model
    MODEL = importlib.import_module(module_name)

    discriminator,generator = MODEL.get_model(NB_GENRES = opt.NB_GENRES,
                                                noise_len= opt.noise_len,
                                                len_seq= opt.len_seq,
                                                nb_notes=opt.nb_notes,
                                                kernal =opt.kernal)
    # generator.apply(weight_init)
    # discriminator.apply(weight_init)

    f_loss = MODEL.get_loss(cuda = opt.cuda)

    print('generator and discriminator models are successfully loaded')

    print('----------------------Model Info----------------------')
    print('Root of prestrain model: ', '[No Prestrained loaded]')
    print('Model: ', opt.model)
    print('Trained Date: ',opt.time)

    print('----------------------Configure optimizer and scheduler----------------------')
    experiment_dir = Path('ckpt/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(opt.model+"_"+opt.kernal)
    experiment_dir.mkdir(exist_ok=True)
    shutil.copy('model/%s.py' % opt.model, str(experiment_dir))
    shutil.move(os.path.join(str(experiment_dir), '%s.py'% opt.model), 
                os.path.join(str(experiment_dir), 'model.py'))
    experiment_dir = experiment_dir.joinpath('saves')
    experiment_dir.mkdir(exist_ok=True)
    opt.save_root = str(experiment_dir)

    print('APEX: ',opt.apex)
    print('CUDA: ',opt.cuda)

    if(opt.apex==True):#APEX not is implmented yet

        model = apex.parallel.convert_syncbn_model(model)
        generator.cuda()
        discriminator.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
        model = torch.nn.DataParallel(model,device_ids =[0,1])

    else:
        if(opt.cuda):
            discriminator.cuda()        
            generator.cuda()

            optimizer_d = optim.Adam(discriminator.parameters(), lr=opt.d_init_lr,betas=(0.9, 0.999),eps=1e-07)
            optimizer_g = optim.Adam(generator.parameters(), lr=opt.g_init_lr,betas=(0.9, 0.999),eps=1e-07)
            scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=opt.d_step_size,gamma = opt.d_gamma)
            scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=opt.g_step_size,gamma = opt.g_gamma)

            if(opt.num_gpu>1):
                discriminator = torch.nn.DataParallel(discriminator)            
                generator = torch.nn.DataParallel(generator)
        else:
            discriminator.cpu()        
            generator.cpu()

            optimizer_d = optim.Adam(discriminator.parameters(), lr=opt.d_init_lr,betas=(0.9, 0.999),eps=1e-07)
            optimizer_g = optim.Adam(generator.parameters(), lr=opt.g_init_lr,betas=(0.9, 0.999),eps=1e-07)
            scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=opt.d_step_size,gamma = opt.d_gamma)
            scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=opt.g_step_size,gamma = opt.g_gamma)

    return opt,generator,discriminator,f_loss,optimizer_d,scheduler_d,optimizer_g,scheduler_g


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def main():
    # setSeed(10)
    opt = opt_global_inti()
    if(opt.cuda):
        opt.device = 'cuda'
    else:
        opt.device = 'cpu'
    if(opt.cuda):
        num_gpu = torch.cuda.device_count()
        print("num gpu avaible:",num_gpu)
        print("opt.num_gpu :",opt.num_gpu )
        assert num_gpu == opt.num_gpu,"opt.num_gpu NOT equals torch.cuda.device_count()" 

    if(opt.wandb):
        import wandb

    print('----------------------Load Dataset----------------------')
    print('Root of dataset: ', opt.dataset_root)
    print('debug: ', opt.debug)

    drum_dataset = drum_dataloader(root = opt.dataset_root)
    print("length of Data: ",len(drum_dataset))
    print('----------------------Music Info----------------------')
    print("DRUM_CLASSES:", DRUM_CLASSES)
    print("# of drum instruments:", nb_notes)
    print("dimentionality of random input z:", len_input)
    print("resolution of one bar:", resolution*4) # how many grids in one bar
    print("length of rhythm pattern to be generated:", len_seq, "beats" )
    opt.DRUM_CLASSES = DRUM_CLASSES
    opt.nb_notes = nb_notes
    opt.len_input = len_input
    opt.resolution = resolution*4
    opt.len_seq = len_seq
    GENRES = np.load(opt.dataset_root)['genres']
    # GENRES:['breakbeat' 'dnb' 'downtempo' 'garage' 'house' 'jungle' 'old_skool' 'techno' 'trance']

    opt.NB_GENRES = len(GENRES)

    if(opt.load_pretrain!=''):
        opt,generator,discriminator,f_loss,optimizer_d,scheduler_d,optimizer_g,scheduler_g = load_pretrained(opt)#Not implement yet
    else:
        opt,generator,discriminator,f_loss,optimizer_d,scheduler_d,optimizer_g,scheduler_g = creating_new_model(opt)


    print('----------------------Prepareing Training----------------------')
    # metrics_list = ['discriminator_loss','generator_loss','generator_acc','time_complexicity','storage_complexicity']
    metrics_list = ['discriminator_loss','generator_loss']
    manager_test = metrics_manager(metrics_list)
    manager_train = metrics_manager(metrics_list)


    if(opt.wandb):
        wandb.init(project=opt.wd_project,name=opt.model,resume=False)
        if(opt.wandb_history == False):
            best_value = 0
        else:
            temp = wandb.restore('best_model.pth',run_path = opt.wandb_id)#Continue to pretrain [Not implement Yet]
            best_value = torch.load(temp.name)['generator_loss']#Continue to pretrain [Not implement Yet]

        wandb.config.update(opt)
        if opt.epoch_ckpt == 0:
            opt.unsave_epoch = 0
        else:
            opt.epoch_ckpt = opt.epoch_ckpt+1 #Continue to train the pretrain [Not implement Yet]



    train_d = True
    train_g = True

    print("wandb",opt.wandb)
    print("train_d",train_d)
    print("train_g",train_g)
    print("epoch_max",opt.epoch_max)
    print("kernal",opt.kernal)
    print("save_fig",opt.save_fig)
    print("show_fig",opt.show_fig)

    for epoch in range(opt.epoch_ckpt,opt.epoch_max):
        generator.train()
        generator.train()
        manager_train.reset()
        nb_steps = int(len(drum_dataset)/opt.batch_size)
        for _ in range(nb_steps):
            #Train D
            if(train_d):
                m_d_loss = 0.0
                for _ in range(opt.K_unrolled_d):
                    subset_indices = np.random.randint(low = 0, high= len(drum_dataset), size=opt.batch_size)            

                    drum,label = drum_dataset.__getitem__(subset_indices)
                    drum,label = drum.to(opt.device),label.to(opt.device)
                    valid = Variable(torch.FloatTensor(opt.batch_size, 1).fill_(1.0), requires_grad=False).to(opt.device)* 0.9 # one-sided soft labeling
                    fake = Variable(torch.FloatTensor(opt.batch_size, 1).fill_(0.0), requires_grad=False).to(opt.device)

                    optimizer_d.zero_grad()
                    d_pred_real = discriminator(drum,label) 
                    d_loss_real = f_loss(d_pred_real, valid)
                    d_loss_real.backward()
                    optimizer_d.step()

                    optimizer_d.zero_grad()
                    noise = np.random.normal(0.0, 1, size=(opt.batch_size,opt.noise_len))
                    noise = torch.FloatTensor(noise).to(opt.device)
                    drum_fake = generator(noise,label)  
                    d_pred_fake = discriminator(drum_fake.detach(),label) # detach to avoid training G on these labels

                    d_loss_fake = f_loss(d_pred_fake, fake)
                    d_loss_fake.backward()
                    optimizer_d.step()

                    d_loss = (d_loss_fake + d_loss_real) / 2
                    m_d_loss += d_loss.item()

                m_d_loss /= float(opt.K_unrolled_d)
                manager_train.update('discriminator_loss',m_d_loss)

            #Train G
            if(train_g):
                m_g_loss = 0.0
                for p in discriminator.parameters():
                    p.requires_grad = False
                for _ in range(opt.K_unrolled_g):
                    optimizer_g.zero_grad()
                    noise = np.random.normal(0.0, 1, size=(opt.batch_size,opt.noise_len))
                    noise = torch.FloatTensor(noise).to(opt.device)

                    label_random = torch.LongTensor(np.random.randint(0, opt.NB_GENRES, (opt.batch_size,1))).to(opt.device)
                    valid = Variable(torch.FloatTensor(opt.batch_size, 1).fill_(1.0), requires_grad=False).to(opt.device)

                    drum_fake = generator(noise,label_random)
                    d_pred_fake = discriminator(drum_fake,label_random)

                    g_loss = f_loss(d_pred_fake, valid)
                    m_g_loss += g_loss.item()
                    g_loss.backward()
                    optimizer_g.step()
                for p in discriminator.parameters():
                    p.requires_grad = True
                m_g_loss /= float(opt.K_unrolled_g)
                manager_train.update('generator_loss',m_g_loss)



            if train_d and train_g:
                if(m_g_loss != 0 and m_d_loss != 0):
                    if m_g_loss / m_d_loss > MAX_LOSS_RATIO:
                        train_d = False
                #                print ("Pausing D")
                    elif m_d_loss / m_g_loss > MAX_LOSS_RATIO:
                        train_g = False
                #                print ("Pausing G")
                else:
                    if(m_g_loss == 0):
                        train_g = False
                    if(m_d_loss == 0):
                        train_d = False
            else:
                train_d = True
                train_g = True

        # scheduler_d.step()
        # scheduler_g.step()
        summery_dict = manager_train.summary()    
        print("epoch:",epoch)
        print("m_d_loss:",summery_dict['discriminator_loss'],"    m_g_loss:",summery_dict['generator_loss'])
        manager_train.reset()
        #================================Visualize sample==============================================================
        if(epoch%opt.vis_perEpoch == 0):
            noise = np.random.normal(0.0, 1, size=(opt.batch_size,opt.noise_len))
            noise = torch.FloatTensor(noise).to(opt.device)
            label_random = torch.LongTensor(np.random.randint(0, opt.NB_GENRES, (opt.batch_size,1))).to(opt.device)
            drum_generated = generator(noise,label_random)
            drum_generated = drum_generated[0,:,:]
            drum_generated = drum_generated.cpu().data.numpy()

            signature = {'epoch':epoch,'genre':GENRES[label_random[0,0]]}
            plot_drum_matrix(drum_generated,save_fig = opt.save_fig,show_fig = opt.show_fig,signature = signature)

            # play_drum_matrix(drum1)
        #================================Visualize sample==============================================================

        #================================Save the models==============================================================
        if(epoch%opt.save_perEpoch == 0):
            package = dict()
            package['discriminator'] = discriminator.state_dict()
            package['scheduler_d'] = scheduler_d
            package['optimizer_d'] = optimizer_d

            package['generator'] = generator.state_dict()
            package['scheduler_g'] = scheduler_g
            package['optimizer_g'] = optimizer_g

            package['epoch'] = epoch
            opt_temp = vars(opt)
            for k in opt_temp:
                package[k] = opt_temp[k]

            for k in summery_dict:
                package[k] = summery_dict[k]

            save_root = opt.save_root+'/d_loss%.4f_g_loss%.4f_Epoch%s.pth'%(package['discriminator_loss'],package['generator_loss'],package['epoch'])
            torch.save(package,save_root)
        #================================Save the models==============================================================

        if(opt.wandb):
            wandb.log(summery_dict)


        if(opt.debug == True):
            pdb.set_trace()


if __name__ == '__main__':
    main()