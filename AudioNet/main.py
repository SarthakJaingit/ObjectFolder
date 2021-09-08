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

'''
modes = np.load(opt.modes_file_path)
N = modes.shape[0]
spec_comps = np.zeros((N, 3, 2, opt.stft_freq_dim, opt.stft_time_dim))
print("number of vertices: ", N)
for i in range(N):
    for j in range(3):
        spec_comps[i, j] = generate_spectrogram_complex(modes[i, j], opt.window_size, opt.hop_size, opt.n_fft)
'''

spec_comps = np.load(opt.specs_file_path)

#N: number of data
#D: number of dimension
#C: number of channels, real and img
#F: number of frequency
#T: number of timestamps
N, D, C, F, T = spec_comps.shape

#normalize spec_comps to [-1, 1]
spec_comps_f1_min = spec_comps[:, 0, :, :, :].min()
spec_comps_f1_max = spec_comps[:, 0, :, :, :].max()
spec_comps_f2_min = spec_comps[:, 1, :, :, :].min()
spec_comps_f2_max = spec_comps[:, 1, :, :, :].max()
spec_comps_f3_min = spec_comps[:, 2, :, :, :].min()
spec_comps_f3_max = spec_comps[:, 2, :, :, :].max()
print("F1 direction min: ", spec_comps_f1_min)
print("F1 direction max: ", spec_comps_f1_max)
print("F2 direction min: ", spec_comps_f2_min)
print("F2 direction max: ", spec_comps_f2_max)
print("F3 direction min: ", spec_comps_f3_min)
print("F3 direction max: ", spec_comps_f3_max)
normalizer_dic = {'f1_min': spec_comps_f1_min, 'f1_max': spec_comps_f1_max, \
            'f2_min': spec_comps_f2_min, 'f2_max': spec_comps_f2_max, \
            'f3_min': spec_comps_f3_min, 'f3_max': spec_comps_f3_max}

#spec_comps_max, spec_comps_min = 0.06, -0.06
spec_comps_x = spec_comps[:, 0, :, :, :]
spec_comps_y = spec_comps[:, 1, :, :, :]
spec_comps_z = spec_comps[:, 2, :, :, :]
#spec_comps_x = 2 * ((spec_comps_x - spec_comps_f1_min) / spec_comps_f1_max) - 1
#spec_comps_y = 2 * ((spec_comps_y - spec_comps_f2_min) / spec_comps_f2_max) - 1
#spec_comps_z = 2 * ((spec_comps_z - spec_comps_f3_min) / spec_comps_f3_max) - 1
spec_comps_x = (spec_comps_x - spec_comps_f1_min) / (spec_comps_f1_max - spec_comps_f1_min)
spec_comps_y = (spec_comps_y - spec_comps_f2_min) / (spec_comps_f2_max - spec_comps_f2_min)
spec_comps_z = (spec_comps_z - spec_comps_f3_min) / (spec_comps_f3_max - spec_comps_f3_min)


#initialize frequency and time features
freq_feats = np.repeat(np.repeat(np.arange(F).reshape((F, 1)), T, axis=1).reshape((1, 1, F, T)), N, axis=0)
time_feats = np.repeat(np.repeat(np.arange(T).reshape((1, T)), F, axis=0).reshape((1, 1, F, T)), N, axis=0)

#normalize frequency and time features to [-1, 1]
freq_feats_min = freq_feats.min()
freq_feats_max = freq_feats.max()
time_feats_min = time_feats.min()
time_feats_max = time_feats.max()
#freq_feats = 2 * ((freq_feats - freq_feats_min) / freq_feats_max) - 1
#time_feats = 2 * ((time_feats - time_feats_min) / time_feats_max) - 1
freq_feats = (freq_feats - freq_feats_min) / (freq_feats_max - freq_feats_min)
time_feats = (time_feats - time_feats_min) / (time_feats_max - time_feats_min)

data_x = np.concatenate((freq_feats, time_feats, spec_comps_x), axis=1)
data_y = np.concatenate((freq_feats, time_feats, spec_comps_y), axis=1)
data_z = np.concatenate((freq_feats, time_feats, spec_comps_z), axis=1)
data_x = np.transpose(data_x.reshape((N, C + 2, -1)), axes = [0, 2, 1])
data_y = np.transpose(data_y.reshape((N, C + 2, -1)), axes = [0, 2, 1])
data_z = np.transpose(data_z.reshape((N, C + 2, -1)), axes = [0, 2, 1])

xyz = np.load(opt.vertex_file_path)
xyz = np.repeat(xyz.reshape((N, 1, 3)), F * T, axis=1)

#normalize xyz to [-1, 1]
xyz_min = xyz.min()
xyz_max = xyz.max()
#xyz = 2 * ((xyz - xyz_min) / xyz_max) - 1
xyz = (xyz - xyz_min) / (xyz_max - xyz_min)

#Now concatenate xyz and feats to get final feats matrix as [x, y, z, f, t, real, img] 
data_x = np.concatenate((xyz, data_x), axis=2).reshape((-1, 7))
data_y = np.concatenate((xyz, data_y), axis=2).reshape((-1, 7))
data_z = np.concatenate((xyz, data_z), axis=2).reshape((-1, 7))
#feats, gts = torch.Tensor(data[:, :-2]).to(device), torch.Tensor(data[:, -2:]).to(device)
feats_x, gts_x = data_x[:, :-2], data_x[:, -2:]
feats_y, gts_y = data_y[:, :-2], data_y[:, -2:]
feats_z, gts_z = data_z[:, :-2], data_z[:, -2:]

embed_fn, input_ch = get_embedder(10, 0)
model = AudioNeRF(D = opt.network_depth, input_ch = input_ch)
model = nn.DataParallel(model).to(opt.device)
print(model)
#grad_vars = list(model.parameters())

if opt.loss_type == "L2":
    loss_fn = torch.nn.MSELoss(reduction='mean')
else:
    loss_fn = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-4, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999995)

batch_loss_x = []
batch_loss_y = []
batch_loss_z = []
batch_crossmodal_loss = []
start = 0
N_iter = opt.iterations
N_rand = opt.batchSize
i_batch = 0
total_batches = 0
epoch = 0
rand_idx = np.random.permutation(feats_x.shape[0])
print("Number of samples per epoch:", feats_x.shape[0])

for i in range(start, N_iter):
    curr_idx = rand_idx[i_batch:i_batch+N_rand]
    curr_feats_x, curr_gts_x = torch.Tensor(feats_x[curr_idx]).to(opt.device), torch.Tensor(gts_x[curr_idx]).to(opt.device)
    curr_feats_y, curr_gts_y = torch.Tensor(feats_y[curr_idx]).to(opt.device), torch.Tensor(gts_y[curr_idx]).to(opt.device)
    curr_feats_z, curr_gts_z = torch.Tensor(feats_z[curr_idx]).to(opt.device), torch.Tensor(gts_z[curr_idx]).to(opt.device)
    embedded_x = embed_fn(curr_feats_x)
    embedded_y = embed_fn(curr_feats_y)
    embedded_z = embed_fn(curr_feats_z)
    results_x, results_y, results_z = model(embedded_x, embedded_y, embedded_z)
    loss_x = loss_fn(results_x, curr_gts_x)
    loss_y = loss_fn(results_y, curr_gts_y) 
    loss_z = loss_fn(results_z, curr_gts_z)
    batch_loss_x.append(loss_x.item())
    batch_loss_y.append(loss_y.item())
    batch_loss_z.append(loss_z.item())

    loss = loss_x + loss_y + loss_z 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    i_batch += N_rand
    total_batches += 1

    if i_batch >= feats_x.shape[0]:
        print("Shuffle data after an epoch!")
        #rand_idx = torch.randperm(feats.shape[0])
        rand_idx = np.random.permutation(feats_x.shape[0])
        i_batch = 0
        epoch += 1
        torch.save({'model_state_dict': model.state_dict(), 'normalizer': normalizer_dic}, os.path.join('.', opt.checkpoints_dir, opt.name, 'model.pt'))
        print("Saving latest model!")
        
    if(total_batches % opt.display_freq == 0):
        print('Display training progress at (epoch %d, total_batches %d)' % (epoch, total_batches))
        avg_loss_x = sum(batch_loss_x)/len(batch_loss_x)
        avg_loss_y = sum(batch_loss_y)/len(batch_loss_y)
        avg_loss_z = sum(batch_loss_z)/len(batch_loss_z)
        total_loss = avg_loss_x + avg_loss_y + avg_loss_z
        print('x direction loss: %.8f, y direction loss: %.8f, z direction loss: %.8f, total loss: %.8f' \
            % (avg_loss_x, avg_loss_y, avg_loss_z, total_loss))
        batch_loss_x = []
        batch_loss_y = []
        batch_loss_z = []
        
        if opt.tensorboard:
            writer.add_scalar('data/loss_x', avg_loss_x, i)
            writer.add_scalar('data/loss_y', avg_loss_y, i)
            writer.add_scalar('data/loss_z', avg_loss_z, i)
            writer.add_scalar('data/total_loss', total_loss, i)
            print('end of display \n')
