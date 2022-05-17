import streamlit as st
import numpy as np
import torch

import components.perturbations.lidar.noise.functional as noise_fn
import components.perturbations.lidar.adversarial.functional as adv_fn
import components.perturbations.lidar.semantic.functional as semantic_fn

def apply_all_perturbations(): ## hack to resolve lidar issue
	apply_perts()
	apply_perts()

def apply_perts():
	perturbed_lidar_point_cloud = st.session_state.original_lidar_point_cloud.copy()
	print('In lidar pert functional original: ', np.amax(perturbed_lidar_point_cloud), np.amin(perturbed_lidar_point_cloud))

	semantic_settings = st.session_state['lidar_perturbation_settings'].get('hallucination', None)
	if semantic_settings is not None:
		perturbed_lidar_point_cloud = semantic_fn.apply_hallucination(input_point_cloud=perturbed_lidar_point_cloud, settings=semantic_settings)

	noise_settings = st.session_state['lidar_perturbation_settings'].get('noise', None)
	if noise_settings is not None:
		perturbed_lidar_point_cloud = noise_fn.apply_noise(input_point_cloud=perturbed_lidar_point_cloud, settings=noise_settings)

	print('In lidar pert functional: ', np.amax(perturbed_lidar_point_cloud), np.amin(perturbed_lidar_point_cloud))
	perturbed_lidar = st.session_state.carla_leaderboard_evaluator.agent_instance.preprocess_lidar(st.session_state.data, perturbed_lidar_point_cloud)

	adversarial_settings = st.session_state['lidar_perturbation_settings'].get('adversarial', None)
	if adversarial_settings is not None:
		perturbed_lidar = adv_fn.apply_adversarial_attack(input_lidar=perturbed_lidar, settings=adversarial_settings)
	#print('Adversarial attack on lidar result: ', perturbed_lidar.shape)

	if torch.is_tensor(perturbed_lidar):
		perturbed_lidar = perturbed_lidar.detach().cpu().numpy().squeeze(0)
		perturbed_lidar = np.moveaxis(perturbed_lidar, 0, 2)

	st.session_state.perturbed_lidar = perturbed_lidar

def reset_all_perturbations():
	st.session_state.lidar_perturbation_settings = {}
	st.session_state.perturbed_lidar = st.session_state.original_lidar