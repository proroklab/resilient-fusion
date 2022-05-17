import streamlit as st

import os
import random
import numpy as np

from PIL import Image

def update_perturbation_settings():
	x = st.session_state['image_hallucination_x']
	y = st.session_state['image_hallucination_y']
	size = st.session_state['image_hallucination_size']

	if (isinstance(size, tuple) and size[1] == 0.0) or (isinstance(size, float) and size == 0.0):
		st.session_state['image_perturbation_settings'].pop('hallucination', None)
		if len(st.session_state['image_perturbation_settings']) == 0:
			st.session_state.perturbed_image = st.session_state.original_image
		return

	st.session_state['image_perturbation_settings']['hallucination'] = {'x': x, 'y': y, 'size': size}

def apply_hallucination(input_image, settings):
	perturbed_image = add_jeep_hallucination(input_image, settings['x'], settings['y'], settings['size'])
	return perturbed_image

def add_jeep_hallucination(input_image, x, y, size):
	if isinstance(size, tuple):
		size = random.uniform(a=size[0], b=size[1])
	if size == 0.0:
		return input_image

	if isinstance(x, tuple):
		x = random.uniform(a=x[0], b=x[1])

	if isinstance(y, tuple):
		y = random.uniform(a=y[0], b=y[1])

	hallucination_im = Image.open('{}/components/perturbations/image/semantic/assets/carla_jeep.png'.format(os.environ['TOOLKIT_FRONTEND']))
	hallucination_im_width, hallucination_im_height = hallucination_im.size

	hallucination_im_width = int(hallucination_im_width * size)
	hallucination_im_height = int(hallucination_im_height * size)

	if hallucination_im_width == 0.0 or hallucination_im_height == 0.0:
		return input_image

	hallucination_im = hallucination_im.resize((hallucination_im_width, hallucination_im_height), Image.ANTIALIAS)
	hallucination_im.convert('RGBA')

	perturbed_im = Image.fromarray(input_image)
	perturbed_im.convert('RGBA')
	input_im_width, input_im_height = perturbed_im.size

	top_x = int(x * input_im_width) - (hallucination_im_width//2)
	top_y = int(y * input_im_height) - (hallucination_im_height//2)

	
	perturbed_im.paste(hallucination_im, (top_x, top_y), hallucination_im)
	perturbed_im.convert('RGB')

	return np.asarray(perturbed_im)
