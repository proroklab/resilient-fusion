import streamlit as st
import numpy as np
import torch

def set_perform_inference_flag():
	st.session_state.perform_inference_flag = True

def prepare_lidar_for_inference(lidar):
	lidar = np.moveaxis(lidar, 2, 0)
	#print('In prepare lidar for inf moveaxis: ', lidar.shape)
	lidar = torch.from_numpy(lidar).unsqueeze(0)
	#print('In prepare lidar for inf tensor unsqueeze: ', lidar.shape)
	lidar = lidar.to('cuda', dtype=torch.float32)
	return lidar

def prepare_image_for_inference(rgb):
	rgb = np.moveaxis(rgb, 2, 0)
	#print('In prepare rgb for inf moveaxis: ', rgb.shape)
	rgb = torch.from_numpy(rgb).unsqueeze(0)
	#print('In prepare rgb for inf tensor unsqueeze: ', rgb.shape)
	rgb = rgb.to('cuda', dtype=torch.float32)
	return rgb

def get_perturbed_tick_data():
	perturbed_data = st.session_state.data.copy()
	if perturbed_data.get('rgb_preprocessed', None) is not None:
		perturbed_data['rgb_preprocessed'] = prepare_image_for_inference(st.session_state.perturbed_image)
	if perturbed_data.get('lidar_preprocessed', None) is not None:
		perturbed_data['lidar_preprocessed'] = prepare_lidar_for_inference(st.session_state.perturbed_lidar)
	return perturbed_data


def perform_carla_inference_step():
	perturbed_data = get_perturbed_tick_data()
	st.session_state.carla_leaderboard_evaluator.run_step(perturbed_data)

def agent_performance_clean_vs_noisy_data():
	agent = st.session_state.carla_leaderboard_evaluator.agent_instance

	clean_data = st.session_state.data.copy()
	perturbed_data = get_perturbed_tick_data()

	#print('Img diff: ', clean_data['rgb_preprocessed']-perturbed_data['rgb_preprocessed'])

	t = m = 0
	for c, n in zip(clean_data['rgb_preprocessed'].flatten(), perturbed_data['rgb_preprocessed'].flatten()):
		if c != n:
			m += 1
		t += 1
	print('Mismatch: {}/{}'.format(m, t))


	pred_clean, control_clean = agent.run_step(tick_data=clean_data, timestamp=None, get_pred=True)
	pred_noisy, control_noisy = agent.run_step(tick_data=perturbed_data, timestamp=None, get_pred=True)

	st.warning('Clean pred: {}, Noisy pred: {}, Clean control: {}, Noisy control: {}'.format(pred_clean, pred_noisy, control_clean, control_noisy))