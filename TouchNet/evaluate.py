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

checkpoint = torch.load(opt.model_path)

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
model = NeRF(D = opt.network_depth, input_ch = input_ch, output_ch = 3)
state_dic = checkpoint["model_state_dict"]
state_dic = strip_prefix_if_present(state_dic, 'module.')
model.load_state_dict(state_dic)
model = nn.DataParallel(model).to(opt.device)
print(model)
model.eval()
loss_fn = torch.nn.MSELoss(reduction='mean')

preds = np.zeros((feats.shape[0], 3))
N_rand = opt.batchSize

batch_loss = []
for i in range(feats.shape[0] // N_rand + 1):
    curr_feats = torch.Tensor(feats[i*N_rand:(i+1)*N_rand]).to(opt.device)
    embedded = embed_fn(curr_feats)
    results = model(embedded)
        
    preds[i*N_rand:(i+1)*N_rand, :] = results.detach().cpu().numpy()
    if i % 100 == 0:
        print("Iter: {}".format(i))

preds = (((preds + 1) / 2) * 255)
preds = np.transpose(preds.reshape((N, -1, 3)), axes = [0, 2, 1]).reshape((N, C, W, H))

preds = np.clip(np.rint(preds), 0, 255)

print(preds.shape)
loss = np.linalg.norm(touch_images - preds) / preds.shape[0]

print("Loss: ", loss)

preds = preds.transpose(0,2,3,1)

#save evaluation results
np.save(os.path.join(opt.results_path), preds)