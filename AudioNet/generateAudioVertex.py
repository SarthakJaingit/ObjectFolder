#!/usr/bin/env python
import argparse
import numpy as np
import os
from scipy.spatial import KDTree
import random
from scipy.io.wavfile import write
import librosa

def transform_mesh_collision_binvox(coordinates, translate, scale):
	coordinates[0] = (coordinates[0] + translate[0]) * scale
	coordinates[1] = (coordinates[1] + translate[1]) * scale
	coordinates[2] = (coordinates[2] + translate[2]) * scale
	return coordinates

# python generateAudioVertex.py --voxel_vertex_file_path /viscam/u/rhgao/datasets/ObjectFiles/YCB/ycb/024_bowl/google_16k/vertices.npy --specs_file_path /viscam/u/rhgao/datasets/ObjectFiles/YCB/ycb/024_bowl/google_16k/specs.npy --obj_verts_file_path /viscam/u/rhgao/datasets/ObjectFiles/YCB/ycb/024_bowl/google_16k/TouchData/verts_forces.npy --binvox_file_path /viscam/u/rhgao/datasets/ObjectFiles/YCB/ycb/024_bowl/google_16k/024_bowl.binvoxlog

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--voxel_vertex_file_path', type=str, required=True)
	parser.add_argument('--specs_file_path', type=str, required=True)
	parser.add_argument('--obj_verts_file_path', type=str, required=True)
	parser.add_argument('--binvox_file_path', type=str, required=True)
	parser.add_argument('--num_audios_to_generate', type=int, default=500)
	parser.add_argument('--output_dir', type=str, required=True)
	args = parser.parse_args()

	random.seed(0)

	dim = 32
	obj_verts = np.load(args.obj_verts_file_path)[:,:3]
	voxel_vertex = np.load(args.voxel_vertex_file_path)
	vert_tree = KDTree(voxel_vertex)
	specs = np.load(args.specs_file_path)

	f = open(args.binvox_file_path, 'r')
	for line in f:
		if 'translate' in line:
			line = line.strip(',')[17:].split(', ')
			translation = line[0:3]
			translation = [float(x) for x in translation]
			scale = float(line[4].strip('\'')[10:])
			print(translation)
			print(scale)

	selected_index = np.random.choice(obj_verts.shape[0], args.num_audios_to_generate, replace=True)
	for j,index in enumerate(selected_index):
		obj_coordinates = obj_verts[index]
		binvox_coordinates = transform_mesh_collision_binvox(obj_coordinates, translation, scale)
		coordinates_in_voxel = binvox_coordinates * dim

		k = 4
		vert_dist = vert_tree.query(coordinates_in_voxel, k)[0]
		voxel_verts_index = vert_tree.query(coordinates_in_voxel, k)[1]

		force_x = random.randint(-5, 5)
		force_y = random.randint(-5, 5)
		force_z = random.randint(-5, 5)
		signal = np.zeros(16000 * 2)
		for i in range(k):
			spec_x = specs[voxel_verts_index[i], 0, 0, :, :] + specs[voxel_verts_index[i], 0, 1, :, :] * 1j
			vertex_modes_signal_x = librosa.istft(spec_x, hop_length=160, win_length=400, length=16000*2)
			spec_y = specs[voxel_verts_index[i], 1, 0, :, :] + specs[voxel_verts_index[i], 1, 1, :, :] * 1j
			vertex_modes_signal_y = librosa.istft(spec_y, hop_length=160, win_length=400, length=16000*2)
			spec_z = specs[voxel_verts_index[i], 2, 0, :, :] + specs[voxel_verts_index[i], 2, 1, :, :] * 1j
			vertex_modes_signal_z = librosa.istft(spec_z, hop_length=160, win_length=400, length=16000*2)
			temp = vertex_modes_signal_x * force_x + vertex_modes_signal_y * force_y + vertex_modes_signal_z * force_z
			signal = signal + temp
			
		signal = signal / np.abs(signal).max()
		# Write WAV file
		output_path = os.path.join(args.output_dir, str(j+1) + '.wav')
		print(output_path)
		write(output_path, 16000, signal.astype(np.float32))






if __name__ == '__main__':
	main()
