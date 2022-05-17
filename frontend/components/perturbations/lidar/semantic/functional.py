import streamlit as st 

import sys
import random
import numpy as np
import pandas as pd

import components.perturbations.lidar.utils as lidar_perturbations_utils

def update_perturbation_settings():
	hallucination_dist = st.session_state['lidar_hallucination_distance']
	hallucination_position = st.session_state['lidar_hallucination_position']
	hallucination_width = st.session_state['lidar_hallucination_width']
	hallucination_height = st.session_state['lidar_hallucination_height']

	if ((isinstance(hallucination_width, tuple) and hallucination_width[1] == 0.0) or (isinstance(hallucination_width, float) and hallucination_width == 0.0)) or ((isinstance(hallucination_height, tuple) and hallucination_height[1] == 0.0) or (isinstance(hallucination_height, float) and hallucination_height == 0.0)):
		st.session_state['lidar_perturbation_settings'].pop('hallucination', None)
		if len(st.session_state['lidar_perturbation_settings']) == 0:
			st.session_state['perturbed_lidar'] = st.session_state['original_lidar']
		return

	st.session_state['lidar_perturbation_settings']['hallucination'] = {'dist': hallucination_dist, 'pos': hallucination_position, 'width': hallucination_width, 'height': hallucination_height}

def apply_hallucination(input_point_cloud, settings):
	perturbed_point_cloud = add_cube_hallucination(input_point_cloud, settings['dist'], settings['pos'], settings['width'], settings['height'])
	return perturbed_point_cloud

def add_cube_hallucination(input_point_cloud, dist, pos, width, height):
	if isinstance(dist, tuple):
		dist = random.uniform(a=dist[0], b=dist[1])

	if isinstance(pos, tuple):
		pos = random.uniform(a=pos[0], b=pos[1])

	if isinstance(width, tuple):
		width = random.uniform(a=width[0], b=width[1])
	if width == 0.0:
		return input_point_cloud

	if isinstance(height, tuple):
		height = random.uniform(a=height[0], b=height[1])
	if height == 0.0:
		return input_point_cloud

	width /= 2 

	print('In lidar semantic functional original PC: ', np.amax(input_point_cloud), np.amin(input_point_cloud))

	x, y, z = input_point_cloud[:, 0].copy(), input_point_cloud[:, 1].copy(), input_point_cloud[:, 2].copy()
	print('Lidar hallucination Input xyz: ', x.shape, y.shape, z.shape)

	r, az, el = lidar_perturbations_utils.cart2sph(x, y, z)

	#r_range = (np.amin(r), np.amax(r))
	r_range = (np.amin(r), min(np.amax(r), 30))
	az_range = (np.amin(az), np.amax(az))
	el_range = (np.amin(el), np.amax(el))
	print('Ranges: ', r_range, az_range, el_range)
	print('Range diffs: ', r_range[1]-r_range[0], az_range[1]-az_range[1], el_range[1]-el_range[0])

	box_r = (dist * (r_range[1] - r_range[0])) + r_range[0]
	box_pos = (pos * (az_range[1] - az_range[0])) + az_range[0]
	box_left = box_pos - width
	box_right = box_pos + width
	box_bottom = el_range[0]
	box_top = box_bottom + (height * (el_range[1] - el_range[0]))
	print('Box dims: ', box_left, box_right,  box_bottom, box_top, box_r, box_pos)

	r_pert = []
	az_pert = []
	el_pert = []

	for i, row in enumerate(zip(r, az, el)):
		if not (box_left<=row[1]<=box_right and box_bottom<=row[2]<=box_top and row[0] >= box_r):
			r_pert.append(row[0])
			az_pert.append(row[1])
			el_pert.append(row[2])

	r_pert += list(np.random.normal(0, 0, 100) + box_r)
	az_pert += list(np.random.uniform(low=box_left, high=box_right, size=(100)))
	el_pert += list(np.random.uniform(low=box_bottom, high=box_top, size=(100)))

	r_pert = np.array(r_pert)
	az_pert = np.array(az_pert)
	el_pert = np.array(el_pert)
	print('New polar dims: ', r_pert.shape, az_pert.shape, el_pert.shape)
	
	x_pert, y_pert, z_pert = lidar_perturbations_utils.sph2cart(r_pert, az_pert, el_pert)
	print('New dims: ', x_pert.shape, y_pert.shape, z_pert.shape)

	perturbed_point_cloud = np.dstack((x_pert, y_pert, z_pert))
	perturbed_point_cloud = perturbed_point_cloud.squeeze(axis=0)
	print('NEW PC: ', perturbed_point_cloud.shape, input_point_cloud.shape)

	return perturbed_point_cloud
	







'''
	if st.session_state.original_lidar is not None:
		import components.perturbations.lidar.noise.functional as lid_fn
		import matplotlib.pyplot as plt
		import numpy as np

		
		pc = st.session_state.original_lidar_point_cloud.copy()
		x, y, z = pc[:, 0].copy(), pc[:, 1].copy(), pc[:, 2].copy()
		print('Lidar Noise Input x, y, z shape: ', x.shape, y.shape, z.shape)

		r, az, el = lid_fn.cart2sph(x, y, z)
		print('Lidar Noise Input r, a, e shape: ', r.shape, az.shape, el.shape)

		az_min = np.amin(az)
		az_max = np.amax(az)
		el_min = np.amin(el)
		el_max = np.amax(el)

		print('PC mins: ', az_min, az_max, el_min, el_max)

		#origin = (0, np.amax(az), np.amin(el)) ## (0 radius, az points move from left to right, el points have fov up and fov down)
		#print('Origin: ', origin)

		#box_left = origin[1]
		#box_right = box_left - az_box
		#box_bottom = origin[2]
		#box_top = box_bottom + el_box
		#print('Box points: ', box_left, box_right, box_bottom, box_top)

		box_w_half = (az_max + az_min)/2
		box_left = box_w_half + 0.1
		box_right = box_w_half - 0.1
		box_bottom = el_min
		box_top = box_bottom + 0.5
		box_r = 10

		print('BOX dims: ', box_w_half, box_left, box_right, box_bottom, box_top)

		r_new = []
		az_new = [] 
		el_new = []

		for i, d in enumerate(zip(r, az, el)):
			if not (box_right<=d[1]<=box_left and box_bottom<=d[2]<=box_top and d[0] >= box_r): 
				r_new.append(d[0])
				az_new.append(d[1])
				el_new.append(d[2])

		r_new += list(np.random.normal(0, 0, 100) + box_r)
		az_new += list(np.random.uniform(low=box_left, high=box_right, size=(100)))
		el_new += list(np.random.uniform(low=box_bottom, high=box_top, size=(100)))

		r_new = np.array(r_new)
		az_new = np.array(az_new)
		el_new = np.array(el_new)
		print('New polar dims: ', r_new.shape, az_new.shape, el_new.shape)
		x_new, y_new, z_new = lid_fn.sph2cart(r_new, az_new, el_new)
		print('New dims: ', x_new.shape, y_new.shape, z_new.shape)
		pc_new = np.dstack((x_new, y_new, z_new))
		pc_new = pc_new.squeeze(axis=0)
		print('NEW PC: ', pc_new.shape, pc.shape)

		#with open('pc_original.txt', 'a') as f:
		#	for i, d in enumerate(zip(x, y, z)):
		#		f.write('{}, {}, {}, {}\n'.format(i, d[0], d[1], d[2])) 
		#print('DONE writing to file!!')



		#with open('pc.txt', 'a') as f:
		#	for i, d in enumerate(zip(r, az, el)):
		#		f.write('{}, {}, {}, {}\n'.format(i, d[0], d[1], d[2])) 
		#print('DONE writing to file!!')

		data_ui.display_point_cloud(col=st, point_cloud=pc)
		data_ui.display_point_cloud(col=st, point_cloud=pc_new)

		pc_new_im = st.session_state.carla_leaderboard_evaluator.agent_instance.preprocess_lidar(st.session_state.data, pc_new)
		pc_new_im = pc_new_im.detach().cpu().numpy().squeeze(0)
		pc_new_im = np.moveaxis(pc_new_im, 0, 2)
		print('PC new im: ', pc_new_im.shape)
		data_ui.display_image(col=st, image=data_fn.lidar_2darray_to_rgb(pc_new_im))
	'''	