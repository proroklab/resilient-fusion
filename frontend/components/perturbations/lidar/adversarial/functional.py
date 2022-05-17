import streamlit as st

import random
import numpy as np
import torch
import torchattacks

import components.inference.functional as inf_fn

def update_perturbation_settings():
	attack_type = st.session_state['lidar_attack_type'].lower()
	remove_adv_flag = False

	if attack_type == 'fgsm':
		eps = st.session_state['lidar_fgsm_epsilon']
		if (isinstance(eps, tuple) and eps[1] == 0.0) or (isinstance(eps, float) and eps == 0.0):
			remove_adv_flag = True
		
		st.session_state['lidar_perturbation_settings']['adversarial'] = {'type': attack_type, 'epsilon': eps}
	
	elif attack_type == 'pgd':
		eps = st.session_state['lidar_pgd_epsilon'] 
		alpha = st.session_state['lidar_pgd_alpha']
		nb_iter = st.session_state['lidar_pgd_iters']
		if (isinstance(nb_iter, tuple) and nb_iter[1] == 0) or (isinstance(nb_iter, int) and nb_iter == 0):
			remove_adv_flag = True
		st.session_state['lidar_perturbation_settings']['adversarial'] = {'type': attack_type, 'epsilon': eps, 'alpha': alpha, 'iterations': nb_iter}
	
	elif attack_type == 'carlini-wagner':
		confidence = st.session_state['lidar_cw_epsilon']
		box_constraints = st.session_state['lidar_cw_box_constraints']
		nb_iter = st.session_state['lidar_cw_iters']
		if (isinstance(nb_iter, tuple) and nb_iter[1] == 0) or (isinstance(nb_iter, int) and nb_iter == 0):
			remove_adv_flag = True
		st.session_state['lidar_perturbation_settings']['adversarial'] = {'type': attack_type, 'confidence': confidence, 'iterations': nb_iter, 'box constraints': box_constraints}

	if remove_adv_flag:
		st.session_state['lidar_perturbation_settings'].pop('adversarial', None)
		if len(st.session_state['lidar_perturbation_settings']) == 0:
			st.session_state.perturbed_lidar = st.session_state.original_lidar
		return

#@torch.no_grad()
def apply_adversarial_attack(input_lidar, settings):
	le = st.session_state.carla_leaderboard_evaluator
	model = le.agent_instance.get_lidar_encoder_model()

	#rgb = inf_fn.prepare_lidar_for_inference(input_lidar)
	lidar_features = le.agent_instance.get_lidar_features(model, input_lidar)
	#rgb /= 255.
	label = torch.argmax(lidar_features)
	label = torch.LongTensor([label])

	if settings['type'] == 'fgsm':
		perturbed_lidar = add_fgsm_attack(model, input_lidar, label, settings['epsilon'])
	elif settings['type'] == 'pgd':
		perturbed_lidar = add_pgd_attack(model, input_lidar, label, settings['epsilon'], settings['alpha'], settings['iterations'])
	elif settings['type'] == 'carlini-wagner':
		perturbed_lidar = add_cw_attack(model, input_lidar, label, settings['confidence'], settings['iterations'], box_constraints=settings['box constraints'], lr=0.01)

	perturbed_lidar = perturbed_lidar.squeeze(0).permute(1, 2, 0)
	#perturbed_image *= 255.0
	perturbed_lidar = perturbed_lidar.detach().cpu().numpy().astype(np.uint8)

	### only added for testing purposes
	pert_im = inf_fn.prepare_lidar_for_inference(perturbed_lidar)
	perturbed_lidar_features = le.agent_instance.get_lidar_features(model, pert_im)
	perturbed_label = torch.argmax(perturbed_lidar_features)
	perturbed_label = torch.LongTensor([perturbed_label])
	print('Testing testing adv attack: ', label, perturbed_label)
	###

	return perturbed_lidar


def add_fgsm_attack(model, lidar, label, epsilon):
	if isinstance(epsilon, tuple):
		epsilon = random.uniform(a=epsilon[0], b=epsilon[1])
	attack = torchattacks.FGSM(model=model, eps=epsilon)
	perturbed_lidar = attack(lidar, label)
	return perturbed_lidar

def add_pgd_attack(model, lidar, label, epsilon, alpha, nb_iter):
	if isinstance(epsilon, tuple):
		epsilon = random.uniform(a=epsilon[0], b=epsilon[1])
	if isinstance(alpha, tuple):
		alpha = random.uniform(a=alpha[0], b=alpha[1])
	if isinstance(nb_iter, tuple):
		nb_iter = random.randint(a=nb_iter[0], b=nb_iter[1])

	attack = torchattacks.PGD(model=model, eps=epsilon, alpha=alpha, steps=nb_iter)
	with torch.no_grad():
		perturbed_lidar = attack(lidar, label)
	return perturbed_lidar

def add_cw_attack(model, lidar, label, confidence, nb_iter,  box_constraints=1e-4, lr=0.01):
	if isinstance(confidence, tuple):
		confidence = random.uniform(a=confidence[0], b=confidence[1])
	if isinstance(nb_iter, tuple):
		nb_iter = random.randint(a=nb_iter[0], b=nb_iter[1])
	if isinstance(box_constraints, tuple):
		box_constraints = random.uniform(a=box_constraints[0], b=box_constraints[1])
		
	attack = torchattacks.CW(model=model, c=box_constraints, kappa=confidence, steps=nb_iter, lr=lr)
	perturbed_lidar = attack(lidar, label)
	return perturbed_lidar



