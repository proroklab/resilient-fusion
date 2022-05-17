import streamlit as st

import cv2
import random
import numpy as np

def update_perturbation_settings():
	blur_amount = st.session_state['image_blur_amount']

	if (isinstance(blur_amount, tuple) and blur_amount[1] == 0.0) or (isinstance(blur_amount, float) and blur_amount == 0.0):
		st.session_state['image_perturbation_settings'].pop('blur', None)
		if len(st.session_state['image_perturbation_settings']) == 0:
			st.session_state.perturbed_image = st.session_state.original_image
		return

	st.session_state['image_perturbation_settings']['blur'] = {'type': 'motion', 'amount': blur_amount}

def apply_blur(input_image, settings):
	print('Entered img blur: ', input_image.shape)

	perturbed_image = add_motion_blur(input_image, settings['amount'])

	print('IN BLUR: ', input_image.shape, perturbed_image.shape)
	print('DTYPE IN BLUR: ', input_image.dtype, perturbed_image.dtype)

	return perturbed_image


def add_motion_blur(input_image, blur_amount, blur_weight=0.1):
	
	if isinstance(blur_amount, tuple):
		#blur_amount = random.uniform(a=max(blur_amount[0], 0.01), b=blur_amount[1]) ## min of the blur_amount cannot be 0.0
		blur_amount = random.uniform(a=blur_amount[0], b=blur_amount[1])
	print('BLUR AMOUNT: ', blur_amount)
	if blur_amount == 0.0:
		return input_image

	blur_amount = blur_weight * blur_amount
	kernel_size = int(blur_amount * 255)

	kernel_h = np.zeros((kernel_size, kernel_size))
	kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
	kernel_h /= kernel_size
	blurred_image = cv2.filter2D(input_image, -1, kernel_h)

	return blurred_image
