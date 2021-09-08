import os, sys
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from model import *
from utils import *
from options import *

#parse arguments
opt = Options().parse()
opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if opt.tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(comment=opt.name)
else:
    writer = None


touch_images = np.load(opt.touch_file_path).transpose(0,3,1,2)
print(touch_images.shape)

#N: number of data
#C: channels
#W: Width dimension
#H: Height dimension
N, C, W, H = touch_images.shape

#normalize touch images to [-1, 1]
touch_images = 2 * touch_images  / 255 - 1

#initialize frequency and time features
w_feats = np.repeat(np.repeat(np.arange(W).reshape((W, 1)), H, axis=1).reshape((1, 1, W, H)), N, axis=0)
h_feats = np.repeat(np.repeat(np.arange(H).reshape((1, H)), W, axis=0).reshape((1, 1, W, H)), N, axis=0)

#normalize frequency and time features to [-1, 1]
w_feats_min = w_feats.min()
w_feats_max = w_feats.max()
h_feats_min = h_feats.min()
h_feats_max = h_feats.max()
w_feats = 2 * ((w_feats - w_feats_min) / w_feats_max) - 1
h_feats = 2 * ((h_feats - h_feats_min) / h_feats_max) - 1

data_x = np.concatenate((w_feats, h_feats, touch_images), axis=1)
data_x = np.transpose(data_x.reshape((N, C + 2, -1)), axes = [0, 2, 1])

xyz = np.load(opt.vertex_file_path)
xyz = xyz[:,0:3]
xyz = np.repeat(xyz.reshape((N, 1, 3)), W * H, axis=1)

#normalize xyz to [-1, 1]
xyz_min = xyz.min()
xyz_max = xyz.max()
xyz = 2 * ((xyz - xyz_min) / xyz_max) - 1
#xyz = (xyz - xyz_min) / (xyz_max - xyz_min)

#Now concatenate xyz and feats to get final feats matrix as [x, y, z, w, h, r, g, b] 
data= np.concatenate((xyz, data_x), axis=2).reshape((-1, 8))
feats, gts = data[:, :-3], data[:, -3:]

embed_fn, input_ch = get_embedder(10, 0)
model = NeRF(D = opt.network_depth, input_ch = input_ch, output_ch=3)
model = nn.DataParallel(model).to(opt.device)
print(model)
#grad_vars = list(model.parameters())

if opt.loss_type == "L2":
    loss_fn = torch.nn.MSELoss(reduction='mean')
else:
    loss_fn = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-4, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999995)

batch_loss = []
start = 0
N_iter = opt.iterations
N_rand = opt.batchSize
i_batch = 0
total_batches = 0
epoch = 0
rand_idx = np.random.permutation(feats.shape[0])
print("Number of samples per epoch:", feats.shape[0])

for i in range(start, N_iter):
    curr_idx = rand_idx[i_batch:i_batch+N_rand]
    curr_feats, curr_gts = torch.Tensor(feats[curr_idx]).to(opt.device), torch.Tensor(gts[curr_idx]).to(opt.device)
    embedded = embed_fn(curr_feats)
    results = model(embedded)
    loss = loss_fn(results, curr_gts)
    batch_loss.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    i_batch += N_rand
    total_batches += 1

    if i_batch >= feats.shape[0]:
        print("Shuffle data after an epoch!")
        rand_idx = np.random.permutation(feats.shape[0])
        i_batch = 0
        epoch += 1
        torch.save({'model_state_dict': model.state_dict()}, os.path.join('.', opt.checkpoints_dir, opt.name, 'model.pt'))
        print("Saving latest model!")
        
    if(total_batches % opt.display_freq == 0):
        print('Display training progress at (epoch %d, total_batches %d)' % (epoch, total_batches))
        avg_loss = sum(batch_loss)/len(batch_loss)
        print('loss: %.8f' % (avg_loss))
        batch_loss = []
        
        if opt.tensorboard:
            writer.add_scalar('data/loss', avg_loss, i)
            print('end of display \n')