import streamlit as st

import sys
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)

def get_data_from_carla():
	st.session_state.data = st.session_state.carla_leaderboard_evaluator.get_scenario_data(is_first_reading=True)
	#print('Obstacle: ', st.session_state.data['obstacle_detect'])
	if st.session_state.data is None:
		st.session_state.original_image = st.session_state.perturbed_image = None
		st.session_state.original_lidar = st.session_state.perturbed_lidar = None
		return 

	## prefereable to use the 'get' based retrieval on dict since some algos might not use certain 
	## sensor, in which case those values must be set to None
	original_image = st.session_state.data.get('rgb_preprocessed', None)
	if original_image is not None:
		original_image = original_image.detach().cpu().numpy().squeeze(0)
		original_image = np.moveaxis(original_image, 0, 2)
		original_image = original_image.astype(np.uint8)
		#print('In get_data_from_carla original_image: ', original_image.shape)
	st.session_state.original_image = original_image
	st.session_state.perturbed_image = st.session_state.original_image
	
	st.session_state.original_lidar_point_cloud = st.session_state.data.get('lidar', None)
	#print('Lidar point cloud: ', st.session_state.original_lidar_point_cloud)
	#print('Lidar point cloud size: ', st.session_state.original_lidar_point_cloud.shape)
	original_lidar = st.session_state.data.get('lidar_preprocessed', None)
	if original_lidar is not None:
		original_lidar = original_lidar.detach().cpu().numpy().squeeze(0)
		original_lidar = np.moveaxis(original_lidar, 0, 2)
	st.session_state.original_lidar = original_lidar
	st.session_state.perturbed_lidar = st.session_state.original_lidar

	'''
	original_lidar_im = st.session_state.data.get('lidar_preprocessed', None)
	if original_lidar_im is not None:
		original_lidar_im = original_lidar_im[0].detach().cpu().numpy().squeeze(0)
		original_lidar_im = np.moveaxis(original_lidar_im, 0, 2)
	st.session_state.original_lidar_im = original_lidar_im
	st.session_state.perturbed_lidar_im = st.session_state.original_lidar_im

	st.session_state.original_lidar = st.session_state.data.get('lidar', None)
	st.session_state.perturbed_lidar = st.session_state.original_lidar
	'''

def lidar_2darray_to_rgb(array):
	#print('LIDAR TO RGB: ', array.shape)
	W, H, C = array.shape
	assert C == 2
	
	img = np.dstack((array, np.zeros(shape=(W, H, 1))))
	return img
