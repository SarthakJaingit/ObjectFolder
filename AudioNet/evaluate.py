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

modes = np.load(opt.modes_file_path)
N = modes.shape[0]
'''
spec_comps = np.zeros((N, 3, 2, opt.stft_freq_dim, opt.stft_time_dim))
print("number of vertices: ", N)
for i in range(N):
    for j in range(3):
        spec_comps[i, j] = generate_spectrogram_complex(modes[i, j], opt.window_size, opt.hop_size, opt.n_fft)
'''
#N: number of data
#D: number of dimension
#C: number of channels, real and img
#F: number of frequency
#T: number of timestamps
N, D, C, F, T = N, 3, 2, opt.stft_freq_dim, opt.stft_time_dim

checkpoint = torch.load(opt.model_path)
normalizer_dic = checkpoint['normalizer']
spec_comps_f1_min = normalizer_dic['f1_min']
spec_comps_f1_max = normalizer_dic['f1_max']
spec_comps_f2_min = normalizer_dic['f2_min']
spec_comps_f2_max = normalizer_dic['f2_max']
spec_comps_f3_min = normalizer_dic['f3_min']
spec_comps_f3_max = normalizer_dic['f3_max']

'''
spec_comps_x = spec_comps[:, 0, :, :, :]
spec_comps_y = spec_comps[:, 1, :, :, :]
spec_comps_z = spec_comps[:, 2, :, :, :]
#spec_comps_x = 2 * ((spec_comps_x - spec_comps_f1_min) / spec_comps_f1_max) - 1
#spec_comps_y = 2 * ((spec_comps_y - spec_comps_f2_min) / spec_comps_f2_max) - 1
#spec_comps_z = 2 * ((spec_comps_z - spec_comps_f3_min) / spec_comps_f3_max) - 1
spec_comps_x = (spec_comps_x - spec_comps_f1_min) / (spec_comps_f1_max - spec_comps_f1_min)
spec_comps_y = (spec_comps_y - spec_comps_f2_min) / (spec_comps_f2_max - spec_comps_f2_min)
spec_comps_z = (spec_comps_z - spec_comps_f3_min) / (spec_comps_f3_max - spec_comps_f3_min)
'''

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

data_x = np.concatenate((freq_feats, time_feats), axis=1)
data_y = np.concatenate((freq_feats, time_feats), axis=1)
data_z = np.concatenate((freq_feats, time_feats), axis=1)
data_x = np.transpose(data_x.reshape((N, 2, -1)), axes = [0, 2, 1])
data_y = np.transpose(data_y.reshape((N, 2, -1)), axes = [0, 2, 1])
data_z = np.transpose(data_z.reshape((N, 2, -1)), axes = [0, 2, 1])

xyz = np.load(opt.vertex_file_path)
xyz = np.repeat(xyz.reshape((N, 1, 3)), F * T, axis=1)

#normalize xyz to [-1, 1]
xyz_min = xyz.min()
xyz_max = xyz.max()
#xyz = 2 * ((xyz - xyz_min) / xyz_max) - 1
xyz = (xyz - xyz_min) / (xyz_max - xyz_min)

#Now concatenate xyz and feats to get final feats matrix as [x, y, z, f, t, real, img] 
feats_x = np.concatenate((xyz, data_x), axis=2).reshape((-1, 5))
feats_y = np.concatenate((xyz, data_y), axis=2).reshape((-1, 5))
feats_z = np.concatenate((xyz, data_z), axis=2).reshape((-1, 5))
#feats, gts = torch.Tensor(data[:, :-2]).to(device), torch.Tensor(data[:, -2:]).to(device)

embed_fn, input_ch = get_embedder(10, 0)
model = AudioNeRF(D = opt.network_depth, input_ch = input_ch)
state_dic = checkpoint["model_state_dict"]
state_dic = strip_prefix_if_present(state_dic, 'module.')
model.load_state_dict(state_dic)
model = nn.DataParallel(model).to(opt.device)
print(model)
model.eval()
loss_fn = torch.nn.MSELoss(reduction='mean')

preds_x = np.zeros((feats_x.shape[0], 2))
preds_y = np.zeros((feats_y.shape[0], 2))
preds_z = np.zeros((feats_z.shape[0], 2))
N_rand = opt.batchSize

batch_loss = []
for i in range(feats_x.shape[0] // N_rand + 1):
    curr_feats_x = torch.Tensor(feats_x[i*N_rand:(i+1)*N_rand]).to(opt.device)
    curr_feats_y = torch.Tensor(feats_y[i*N_rand:(i+1)*N_rand]).to(opt.device)
    curr_feats_z = torch.Tensor(feats_z[i*N_rand:(i+1)*N_rand]).to(opt.device)
    embedded_x = embed_fn(curr_feats_x)
    embedded_y = embed_fn(curr_feats_y)
    embedded_z = embed_fn(curr_feats_z)
    results_x, results_y, results_z = model(embedded_x, embedded_y, embedded_z)
        
    preds_x[i*N_rand:(i+1)*N_rand, :] = results_x.detach().cpu().numpy()
    preds_y[i*N_rand:(i+1)*N_rand, :] = results_y.detach().cpu().numpy()
    preds_z[i*N_rand:(i+1)*N_rand, :] = results_z.detach().cpu().numpy()
    if i % 100 == 0:
        print("Iter: {}".format(i))

#preds_x = (((preds_x + 1) / 2) * spec_comps_f1_max) + spec_comps_f1_min
#preds_y = (((preds_y + 1) / 2) * spec_comps_f2_max) + spec_comps_f2_min
#preds_z = (((preds_z + 1) / 2) * spec_comps_f3_max) + spec_comps_f3_min
preds_x = preds_x * (spec_comps_f1_max - spec_comps_f1_min) + spec_comps_f1_min
preds_y = preds_y * (spec_comps_f2_max - spec_comps_f2_min) + spec_comps_f2_min
preds_z = preds_z * (spec_comps_f3_max - spec_comps_f3_min) + spec_comps_f3_min
preds_x = np.transpose(preds_x.reshape((N, -1, 2)), axes = [0, 2, 1]).reshape((N, 1, C, F, T))
preds_y = np.transpose(preds_y.reshape((N, -1, 2)), axes = [0, 2, 1]).reshape((N, 1, C, F, T))
preds_z = np.transpose(preds_z.reshape((N, -1, 2)), axes = [0, 2, 1]).reshape((N, 1, C, F, T))
preds = np.concatenate((preds_x, preds_y, preds_z), axis=1)

#save evaluation results
np.save(os.path.join(opt.results_dir, 'specs_pred'), preds)
'''
modes_pred = np.zeros((N, 3, opt.audio_length * opt.audio_sampling_rate))
for i in range(N):
    for j in range(3):
        complex_spec = preds[i, j, 0, :, :] + preds[i, j, 1, :, :] * 1j
        audio = librosa.istft(complex_spec, hop_length=160, win_length=400, length=int(opt.audio_length * opt.audio_sampling_rate))
        print(audio.shape)
        modes_pred[i,j,:] = audio
np.save(os.path.join(opt.results_dir, 'modes_pred'), modes_pred)
'''