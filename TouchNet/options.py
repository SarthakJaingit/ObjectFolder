#!/usr/bin/env python

import argparse
import os
import torch
from utils import *

class Options():
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		self.initialized = False

	def initialize(self):
		self.parser.add_argument('--vertex_file_path', default='./data/vertex.npy', help='path to dataset')
		self.parser.add_argument('--touch_file_path', default='./data/touch.npy', help='path to dataset')
		self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
		self.parser.add_argument('--name', type=str, default='audio-nerf', help='name of the experiment. It decides where to store models')
		self.parser.add_argument('--checkpoints_dir', type=str, default='checkpoints/', help='models are saved here')
		self.parser.add_argument('--batchSize', type=int, default=10000, help='input batch size')
		self.parser.add_argument('--iterations', type=int, default=400000, help='number of batches to train')
		self.parser.add_argument('--nThreads', default=16, type=int, help='# threads for loading data')
		self.parser.add_argument('--seed', default=0, type=int, help='random seed')
		self.parser.add_argument('--tensorboard', type=bool, default=False, help='use tensorboard to visualize loss change ')

		# arguments
		self.parser.add_argument('--rgb_width', default=160, type=int, help="touch image width")
		self.parser.add_argument('--rgb_height', default=120, type=int, help="touch image height")
		self.parser.add_argument('--network_depth', type=int, default=8, help='depth of network')

		# training
		self.parser.add_argument('--display_freq', type=int, default=10, help='frequency of displaying average loss and accuracy')
		self.parser.add_argument('--loss_type', default='L2', type=str, choices=('L1', 'L2'), help='type of loss')

		# testing
		self.parser.add_argument('--model_path', type=str, default='./checkpoints/audio-nerf/model.pt', help='frequency of displaying average loss and accuracy')
		self.parser.add_argument('--results_path', type=str, default='./results/audio-nerf/', help='dir to save evaluation results')

		self.initialized = True

	def parse(self):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()

		str_ids = self.opt.gpu_ids.split(',')
		self.opt.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				self.opt.gpu_ids.append(id)

		# set gpu ids
		if len(self.opt.gpu_ids) > 0:
			torch.cuda.set_device(self.opt.gpu_ids[0])

		#I should process the opt here, like gpu ids, etc.
		args = vars(self.opt)
		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')

		# save to the disk
		expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
		mkdirs(expr_dir)
		file_name = os.path.join(expr_dir, 'opt.txt')
		with open(file_name, 'wt') as opt_file:
			opt_file.write('------------ Options -------------\n')
			for k, v in sorted(args.items()):
				opt_file.write('%s: %s\n' % (str(k), str(v)))
			opt_file.write('-------------- End ----------------\n')
		return self.opt