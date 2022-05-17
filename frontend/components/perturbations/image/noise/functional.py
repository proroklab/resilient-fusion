import streamlit as st

import random
import numpy as np
import skimage.util as sutil

def update_perturbation_settings():
	noise_type = st.session_state['image_noise_type'].lower()
	remove_noise_flag = False

	if noise_type in ['salt', 'pepper', 's&p']:
		amount = st.session_state['image_sp_amount']
		if (isinstance(amount, tuple) and amount[1] == 0.0) or (isinstance(amount, float) and amount == 0.0):
			remove_noise_flag = True
		st.session_state['image_perturbation_settings']['noise'] = {'type': noise_type, 'amount': amount}
	elif noise_type == 'gaussian':
		variance = st.session_state['image_gaussian_variance']
		if (isinstance(variance, tuple) and variance[1] == 0.0) or (isinstance(variance, float) and variance == 0.0):
			remove_noise_flag = True
		st.session_state['image_perturbation_settings']['noise'] = {'type': noise_type, 'variance': variance}

	if remove_noise_flag:
		st.session_state['image_perturbation_settings'].pop('noise', None)
		if len(st.session_state['image_perturbation_settings']) == 0:
			st.session_state.perturbed_image = st.session_state.original_image
		return

def apply_noise(input_image, settings):
	print('Entered img noise: ', input_image.shape)

	if settings['type'] in ['salt', 'pepper', 's&p']:
		perturbed_image = add_salt_pepper_noise(input_image, settings['type'], settings['amount'])
	elif settings['type'] == 'gaussian':
		perturbed_image = add_gaussian_noise(input_image, settings['type'], settings['variance'])
	
	if input_image.dtype == np.uint8 and perturbed_image.dtype in [np.float32, np.float64]:
		perturbed_image *= 255
		perturbed_image = perturbed_image.astype(np.uint8)

	print('IN NOISE: ', input_image.dtype, perturbed_image.dtype)
	return perturbed_image


def add_salt_pepper_noise(input_image, noise_type, amount):
	if isinstance(amount, tuple):
		amount = random.uniform(a=amount[0], b=amount[1])
	return sutil.random_noise(image=input_image, mode=noise_type, amount=amount)

def add_gaussian_noise(input_image, noise_type, variance):
	if isinstance(variance, tuple):
		variance = random.uniform(a=variance[0], b=variance[1])
	return sutil.random_noise(image=input_image, mode=noise_type, var=variance)