#!/usr/bin/env python
import argparse
import numpy as np
import os
import random
from PIL import Image

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--touch_path', type=str, required=True)
	parser.add_argument('--num_touchs_to_generate', type=int, default=500)
	parser.add_argument('--output_dir', type=str, required=True)
	args = parser.parse_args()

	touch = np.load(args.touch_path)

	selected_index = np.random.choice(touch.shape[0], args.num_touchs_to_generate, replace=True)
	for index,i in enumerate(selected_index):
		print(index)
		touch_image = touch[i]
		img = Image.fromarray(touch_image.astype(np.uint8), 'RGB')
		img_save_path = os.path.join(args.output_dir, str(index+1) + '.png')
		img.save(img_save_path)






if __name__ == '__main__':
	main()
