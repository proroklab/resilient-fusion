import streamlit as st

import cv2
import random
import numpy as np

from PIL import Image, ImageDraw

def update_perturbation_settings():
	x = st.session_state['image_occlusion_x']
	y = st.session_state['image_occlusion_y']
	width = st.session_state['image_occlusion_width']
	#colour = st.session_state['image_occlusion_colour']
	colour = 'black'

	if (isinstance(width, tuple) and width[1] == 0.0) or (isinstance(width, float) and width == 0.0):
		st.session_state['image_perturbation_settings'].pop('occlusion', None)
		if len(st.session_state['image_perturbation_settings']) == 0:
			st.session_state.perturbed_image = st.session_state.original_image
		return

	st.session_state['image_perturbation_settings']['occlusion'] = {'x': x, 'y': y, 'width': width, 'colour': colour}

def apply_occlusion(input_image, settings):
	perturbed_image = add_square_occlusion(input_image, settings['x'], settings['y'], settings['width'], settings['colour'])
	return perturbed_image

def add_square_occlusion(input_image, x, y, width, colour):
	if isinstance(width, tuple):
		width = random.uniform(a=width[0], b=width[1])
	if width == 0.0:
		return input_image

	if isinstance(x, tuple):
		x = random.uniform(a=x[0], b=x[1])

	if isinstance(y, tuple):
		y = random.uniform(a=y[0], b=y[1])

	im_length, im_width, im_channel = input_image.shape

	x_pos = int(x * im_width)
	y_pos = int(y * im_length)
	half_s = int(width * min(im_width, im_length))/2

	im = Image.fromarray(input_image)
	draw = ImageDraw.Draw(im)
	draw.rectangle((x_pos - half_s, y_pos - half_s, x_pos + half_s, y_pos + half_s), fill=colour.lower())

	perturbed_im = np.asarray(im)

	return perturbed_im



