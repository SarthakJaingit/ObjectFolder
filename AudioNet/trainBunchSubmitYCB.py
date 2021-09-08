#!/usr/bin/env python
import os
import subprocess
import json
import random
import glob
import h5py
import argparse
import pickle
from utils import *

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

#python trainBunchSubmitYCB.py --data_root /viscam/u/rhgao/datasets/ObjectFiles/YCB/ycb/ --data_csv /viscam/u/rhgao/datasets/ObjectFiles/YCB.csv --num_of_jobs_to_submit 2

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_root', type=str, required=True)
	parser.add_argument('--data_csv', type=str, required=True)
	parser.add_argument('--num_of_jobs_to_submit', type=int, default=1, help="num of jobs to submit")
	parser.add_argument('--job_index_start', type=int, default=0, help="num of classes")
	args = parser.parse_args()

	data_csv_file = open(args.data_csv, 'r')
	cmds2execute = []
	for line in data_csv_file:
		parts = line.strip().split(',')
		model_name = parts[1]
		material_type = parts[3]
		length = parts[2]
		model_root = os.path.join(args.data_root, model_name, 'google_16k')
		vertex_file_path = os.path.join(model_root, 'vertices.npy')
		modes_file_path = os.path.join(model_root, 'modes.npy')
		specs_file_path = os.path.join(model_root, 'specs.npy')
		log_file_path = os.path.join('.', 'logs', model_name + '.log')
		model_file = os.path.join(model_root, model_name + '_AudioNetNew', 'model.pt')
		if os.path.exists(model_file):
			print(model_file)
			continue
		cmd = 'python main.py' +  \
			' --vertex_file_path ' + vertex_file_path + \
			' --modes_file_path ' + modes_file_path + \
			' --specs_file_path ' + specs_file_path + \
			' --checkpoints_dir ' + model_root + \
			' --gpu_ids 0,1' + \
			' --display_freq 100' + \
			' --batchSize 50000' + \
			' --iterations 300000' + \
			' --tensorboard True' + \
			' --name ' + model_name + '_AudioNetNew' + ' |& tee ' + log_file_path
		cmd = cmd + '\n'
		cmds2execute.append(cmd)

	job_splits = list(split(range(len(cmds2execute)), args.num_of_jobs_to_submit))
	count = 0
	for sub_split in job_splits:
		count = count + 1
		print(count)
		script2exe = open('slurm_script/' + str(args.job_index_start + count) + '.sh', 'w')
		script2exe.write('#!/bin/bash\n')
		for i in sub_split:
			script2exe.write(cmds2execute[i])
		script2exe.close()
		cmd = 'chmod a+x \'slurm_script/' + str(args.job_index_start + count)  + '.sh\''
		subprocess.call(cmd, shell=True)

		# generate slurm submit file
		slurm_file = open('submit.slurm','w')
		slurm_file.write('#!/bin/bash\n')
		slurm_file.write('#SBATCH --job-name=' + 'job' + str(args.job_index_start + count) + '\n')
		slurm_file.write('#SBATCH --output=' + 'slurm_output/job' + str(args.job_index_start + count) + '.out\n')
		slurm_file.write('#SBATCH --error=' + 'slurm_output/job' + str(args.job_index_start + count) + '.err\n')
		slurm_file.write('#SBATCH --mem=200G\n')
		slurm_file.write('#SBATCH --nodes=1\n')
		slurm_file.write('#SBATCH --ntasks-per-node=1\n')
		slurm_file.write('#SBATCH --time 120:00:00\n')
		slurm_file.write('#SBATCH --partition=svl\n')  #viscam, svl
		slurm_file.write('#SBATCH --gres=gpu:2\n')
		slurm_file.write('#SBATCH --cpus-per-task=10\n')
		slurm_file.write('source /sailhome/rhgao/.bashrc\n')
		slurm_file.write('conda activate nerf\n')
		slurm_file.write('./slurm_script/' + str(args.job_index_start + count) + '.sh\n')
		slurm_file.close()
		cmd = 'sbatch submit.slurm'
		subprocess.call(cmd, shell=True)

if __name__ == '__main__':
	main()
