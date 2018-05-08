import cv2
import numpy as np
import collections as cl
import pandas as pd
import time
from tqdm import tqdm
import os
import yaml
from multiprocessing import Process, Queue, cpu_count


# This list of ascii characters corresponds with the
#	characters in ascii_sprites.jpg
ascii_characters = [
'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
'0','1','2','3','4','5','6','7','8','9',
'!','@','#','$','%','^','&','*','(',')',
'`','~','-','_','=','+','[','{',']','}','\\','|',';',':','\'','\"',',','<','.','>','/','?'
]


def read_in_ascii_sprites(ascii_sprite_file, num_ascii_chars, ascii_char_list, sprite_resize, sprite_dilations):
	ascii_chars = cl.defaultdict(list)
	ascii_sprites = cv2.imread(ascii_sprite_file, 0).astype(np.float64)
	height = ascii_sprites.shape[0]
	width = int(np.ceil(ascii_sprites.shape[1] / float(num_ascii_chars)))
	for i in range(num_ascii_chars):
		current_char = ascii_characters[i]
		if ('all' in set(ascii_char_list)) or (current_char in set(ascii_char_list)):
			char_sprite = np.ones((height, width), dtype=np.float64) * 255
			section = ascii_sprites[:height, i*width:(i+1)*width]
			char_sprite[:section.shape[0], :section.shape[1]] = section
			_, char_sprite = cv2.threshold(char_sprite, 127, 255, cv2.THRESH_BINARY)
			if sprite_resize < 1.0:
				itp = cv2.INTER_AREA
				char_sprite = cv2.resize(char_sprite, (0,0), fx=sprite_resize, fy=sprite_resize, interpolation=itp)
			else:
				itp = cv2.INTER_LINEAR
				char_sprite = cv2.resize(char_sprite, (0,0), fx=sprite_resize, fy=sprite_resize, interpolation=itp)
			if sprite_dilations > 0:
				kernel = np.ones((2,2), np.uint8)
				char_sprite = cv2.erode(char_sprite, kernel, iterations=sprite_dilations)
			elif sprite_dilations < 0:
				kernel = np.ones((2,2), np.uint8)
				char_sprite = cv2.dilate(char_sprite, kernel, iterations=np.abs(sprite_dilations))
			avg_gs = np.mean(char_sprite)
			ascii_chars['sprite'].append(char_sprite)
			ascii_chars['avg_gs'].append(avg_gs)
	imin = np.min(ascii_chars['avg_gs'])
	imax = np.max(ascii_chars['avg_gs'])
	ascii_chars['avg_gs'] = [(255.0*(v-imin))/(imax-imin) for v in ascii_chars['avg_gs']]
	ascii_char_df = pd.DataFrame(ascii_chars)
	final_height = ascii_chars['sprite'][0].shape[0]
	final_width = ascii_chars['sprite'][0].shape[1]
	return ascii_char_df, final_height, final_width


def get_ascii_img(img, ascii_char_df, height, width, sprite_color_thresh, sprite_color):

	def find_nearest(array, value):
		idx = (np.abs(array-value)).argmin()
		return idx

	# Read in input image
	input_img_color = img
	input_img = cv2.cvtColor(input_img_color, cv2.COLOR_BGR2GRAY)
	input_img_color = input_img_color.astype(np.float64)
	input_img = input_img.astype(np.float64)

	# Divide image into cells the size of ascii sprites
	#	and take the average of each
	kernel = np.ones((height, width), np.float64) / (height * width)
    	cell_avgs = cv2.filter2D(input_img, -1, kernel, borderType=cv2.BORDER_REFLECT)[::height, ::width]
	cell_avgs = np.round_(cell_avgs, decimals=0)

	# Divide color image into cells the size of ascii
	#	sprites and take the average of each
	B_avg = cv2.filter2D(input_img_color[:,:,0], -1, kernel, borderType=cv2.BORDER_REFLECT)[::height, ::width]
	G_avg = cv2.filter2D(input_img_color[:,:,1], -1, kernel, borderType=cv2.BORDER_REFLECT)[::height, ::width]
	R_avg = cv2.filter2D(input_img_color[:,:,2], -1, kernel, borderType=cv2.BORDER_REFLECT)[::height, ::width]
	color_avgs = np.dstack((B_avg, G_avg, R_avg))

	# Create look up dictionary for mapping avg
	#	pixel intensities to ascii characters
	look_up = {}
 	for avg_gs in cell_avgs.flatten():
		if avg_gs not in look_up:
			sprite_idx = find_nearest(ascii_char_df['avg_gs'], avg_gs)
			sprite = ascii_char_df.iloc[sprite_idx]['sprite']
			look_up[avg_gs] = sprite

	# Build an ascii art image by iteratively
	#	overlaying ascii characters over 
	#	the original image
	template = input_img_color.copy()
	for i in tqdm(range(cell_avgs.shape[0])):
		for j in range(cell_avgs.shape[1]):
			avg_gs = cell_avgs[i,j]
			sprite = look_up[avg_gs]
			sprite_rows = sprite.shape[0]
			sprite_cols = sprite.shape[1]
			if (i+1)*sprite_rows < input_img.shape[0] and (j+1)*sprite_cols < input_img.shape[1]:
				row_s = i*sprite_rows
				row_f = (i+1)*sprite_rows
				col_s = j*sprite_cols
				col_f = (j+1)*sprite_cols

				# Color the sprites, or not
				if sprite_color:
					color_sprite_shape = (sprite.shape[0], sprite.shape[1], 3)
					color_sprite = np.ones(color_sprite_shape, dtype=sprite.dtype) * 255
					avg_color = color_avgs[i,j]
					color_sprite[sprite <= 255.0 - (sprite_color_thresh * 255.0)] = avg_color
					template[row_s:row_f, col_s:col_f] = color_sprite
				else:
					sprite = cv2.cvtColor(sprite.astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float64)
					template[row_s:row_f, col_s:col_f] = sprite

	# Parse excess rows and columns
	template = template[:row_f, :col_f]

	# Return generated image
	return template.astype(np.uint8)
					
			
def image_pipeline(input_file, img_resize, ascii_sprite_file, num_ascii_chars, ascii_char_list, 
			sprite_dilations, sprite_resize, sprite_color, sprite_color_thresh, filename, file_ext):

	# Read in ascii sprites
	ascii_char_df, height, width = read_in_ascii_sprites(ascii_sprite_file, num_ascii_chars, ascii_char_list, sprite_resize, sprite_dilations)

	# Read in input image
	input_img = cv2.imread(input_file)
	if img_resize < 1.0:
		itp = cv2.INTER_AREA
	else:
		itp = cv2.INTER_LINEAR
	input_img = cv2.resize(input_img, (0,0), fx=img_resize, fy=img_resize, interpolation=itp)
	
	# Create an ascii image
	ascii_image = get_ascii_img(input_img, ascii_char_df, height, width, sprite_color_thresh, sprite_color)

	# Save ascii image
	cv2.imwrite('{}_output{}'.format(filename, file_ext), ascii_image)


def get_ascii_imgs_MP(sub, sub_counter, q, ascii_char_df, height, width, sprite_color_thresh, sprite_color):
	ascii_frames = []
	for frame in tqdm(sub):
		ascii_image = get_ascii_img(frame, ascii_char_df, height, width, sprite_color_thresh, sprite_color)
		ascii_frames.append(ascii_image)
	q.put((sub_counter, ascii_frames))


def video_pipeline(input_file, img_resize, ascii_sprite_file, num_ascii_chars, ascii_char_list, 
			sprite_dilations, sprite_resize, sprite_color, sprite_color_thresh, nth_frames, fps, filename):

	def chunks(l):
		num_cpu = cpu_count()
		n = int(np.ceil(float(len(l)) / num_cpu))
		sub_counter = 0
		for i in range(0, len(l), n):
			yield l[i:i + n], sub_counter
			sub_counter += 1

	# Read in ascii sprites
	ascii_char_df, height, width = read_in_ascii_sprites(ascii_sprite_file, num_ascii_chars, ascii_char_list, sprite_resize, sprite_dilations)

	# Extract frames from video
	frames = []
	cap = cv2.VideoCapture(input_file)
	while(1):
		ret, frame = cap.read()
		if ret:
			frames.append(frame)
		else:
			break
	cap.release()

	# Preprocess frames
	frames_pp = []
	for index in range(len(frames)):
		frame = frames[index]
		if index % nth_frames == 0:
			if img_resize < 1.0:
				itp = cv2.INTER_AREA
			else:
				itp = cv2.INTER_LINEAR
			frame = cv2.resize(frame, (0,0), fx=img_resize, fy=img_resize, interpolation=itp)
			frames_pp.append(frame)
	
	# For each frame, create an ascii image
	# across multiple processors
	q = Queue()
	processes = []
	for sub, sub_counter in chunks(frames_pp):
		p = Process(target=get_ascii_imgs_MP, args=(sub, sub_counter, q, ascii_char_df, height, width, sprite_color_thresh, sprite_color))
		p.start()
		processes.append(p)

	# Construct list of frames from the sub-lists
	# returned from each process
	processed_frames = []
	while True:
		running = any(p.is_alive() for p in processes)
		while not q.empty():
			item = q.get()
			processed_frames.append(item)
		if not running:
			break
	ascii_frames = []
	for (i, frames) in sorted(processed_frames):
		ascii_frames += frames

	# Create video
	fourcc = cv2.VideoWriter_fourcc(*'DIVX')
	shape = (ascii_frames[0].shape[1], ascii_frames[0].shape[0])
	out = cv2.VideoWriter('{}_output.avi'.format(filename), fourcc, fps, shape)
	for frame in ascii_frames:
		out.write(frame)
	out.release()


def get_start_time():
	return time.time()


def get_time_passed(start_time):
	print('--- %s seconds ---' % (time.time() - start_time))


def main():

	# Get start time
	start_time = get_start_time()
	
	# Get variables from configuration file
	with open('config.yaml', 'r') as config:
		cfg = yaml.load(config)
	img_exts = cfg['img_exts']
	vid_exts = cfg['vid_exts']
	ascii_sprite_file = cfg['ascii_sprite_file']
	num_ascii_chars = cfg['num_ascii_chars']
	ascii_char_list = cfg['ascii_char_list']
	sprite_resize = cfg['sprite_resize']
	sprite_dilations = cfg['sprite_dilations']
	sprite_color = cfg['sprite_color']
	sprite_color_thresh = cfg['sprite_color_thresh']
	img_resize = cfg['img_resize']
	nth_frames = cfg['nth_frames']
	fps = cfg['fps']
	input_file = cfg['input_file']

	# Run appropriate pipeline
	filename, file_ext = os.path.splitext(input_file)
	if file_ext in img_exts:
		image_pipeline(input_file, img_resize, ascii_sprite_file, num_ascii_chars, ascii_char_list, 
				sprite_dilations, sprite_resize, sprite_color, sprite_color_thresh, filename, file_ext)
	elif file_ext in vid_exts:
		video_pipeline(input_file, img_resize, ascii_sprite_file, num_ascii_chars, ascii_char_list, 
				sprite_dilations, sprite_resize, sprite_color, sprite_color_thresh, nth_frames, fps, filename)
	else:
		raise Exception('File extension {} not recognized.'.format(file_ext))

	# Print runtime
	get_time_passed(start_time)


if __name__ == '__main__':
	main()
