import streamlit as st

import os
import random
import numpy as np
import torch
import torchattacks

import components.inference.functional as inf_fn
from advGAN.advGan import AdvGAN_Attack

def update_perturbation_settings():
	attack_type = st.session_state['image_attack_type'].lower()
	remove_adv_flag = False

	if attack_type == 'fgsm':
		eps = st.session_state['image_fgsm_epsilon']
		if (isinstance(eps, tuple) and eps[1] == 0.0) or (isinstance(eps, float) and eps == 0.0):
			remove_adv_flag = True
		
		st.session_state['image_perturbation_settings']['adversarial'] = {'type': attack_type, 'epsilon': eps}
	
	elif attack_type == 'pgd':
		eps = st.session_state['image_pgd_epsilon'] 
		alpha = st.session_state['image_pgd_alpha']
		nb_iter = st.session_state['image_pgd_iters']
		if (isinstance(nb_iter, tuple) and nb_iter[1] == 0) or (isinstance(nb_iter, int) and nb_iter == 0):
			remove_adv_flag = True
		st.session_state['image_perturbation_settings']['adversarial'] = {'type': attack_type, 'epsilon': eps, 'alpha': alpha, 'iterations': nb_iter}
	
	elif attack_type == 'carlini-wagner':
		confidence = st.session_state['image_cw_epsilon']
		#binary_search_steps = st.session_state['image_cw_binary_search_steps']
		box_constraints = st.session_state['image_cw_box_constraints']
		nb_iter = st.session_state['image_cw_iters']
		if (isinstance(nb_iter, tuple) and nb_iter[1] == 0) or (isinstance(nb_iter, int) and nb_iter == 0):
			remove_adv_flag = True
		st.session_state['image_perturbation_settings']['adversarial'] = {'type': attack_type, 'confidence': confidence, 'iterations': nb_iter, 'box constraints': box_constraints}

	elif attack_type == 'gan':
		#model_weights_path = '/home/saashanair/toolkit_STABLE/G_epoch_3_300.pth'
		#model_weights_path = '{}/{}'.format(os.environ['GAN_MODELS_ROOT'], st.session_state['image_gan_model_weights'])
		model_weights_path = st.session_state['image_gan_model_weights']
		st.session_state['image_perturbation_settings']['adversarial'] = {'type': attack_type, 'model_weights': model_weights_path}

	if remove_adv_flag:
		st.session_state['image_perturbation_settings'].pop('adversarial', None)
		if len(st.session_state['image_perturbation_settings']) == 0:
			st.session_state.perturbed_image = st.session_state.original_image
		return

def apply_adversarial_attack(input_image, settings):
	le = st.session_state.carla_leaderboard_evaluator
	model = le.agent_instance.get_image_encoder_model()

	rgb = inf_fn.prepare_image_for_inference(input_image)
	image_features = le.agent_instance.get_image_features(model, rgb)
	rgb /= 255.
	label = torch.argmax(image_features)
	label = torch.LongTensor([label])

	if settings['type'] == 'fgsm':
		perturbed_image = add_fgsm_attack(model, rgb, label, settings['epsilon'])
	elif settings['type'] == 'pgd':
		perturbed_image = add_pgd_attack(model, rgb, label, settings['epsilon'], settings['alpha'], settings['iterations'])
	elif settings['type'] == 'carlini-wagner':
		perturbed_image = add_cw_attack(model, rgb, label, settings['confidence'], settings['iterations'], box_constraints=settings['box constraints'], lr=0.01)
	elif settings['type'] == 'gan':
		perturbed_image = add_gan_attack(rgb, settings['model_weights'])

	perturbed_image = perturbed_image.squeeze(0).permute(1, 2, 0)
	perturbed_image *= 255.0
	perturbed_image = perturbed_image.detach().cpu().numpy().astype(np.uint8)

	### only added for testing purposes
	pert_im = inf_fn.prepare_image_for_inference(perturbed_image)
	perturbed_image_features = le.agent_instance.get_image_features(model, pert_im)
	perturbed_label = torch.argmax(perturbed_image_features)
	perturbed_label = torch.LongTensor([perturbed_label])
	print('Testing testing adv attack: ', label, perturbed_label)
	###

	return perturbed_image


def add_fgsm_attack(model, rgb, label, epsilon):
	if isinstance(epsilon, tuple):
		epsilon = random.uniform(a=epsilon[0], b=epsilon[1])
	attack = torchattacks.FGSM(model=model, eps=epsilon)
	perturbed_image = attack(rgb, label)
	return perturbed_image

def add_pgd_attack(model, rgb, label, epsilon, alpha, nb_iter):
	if isinstance(epsilon, tuple):
		epsilon = random.uniform(a=epsilon[0], b=epsilon[1])
	if isinstance(alpha, tuple):
		alpha = random.uniform(a=alpha[0], b=alpha[1])
	if isinstance(nb_iter, tuple):
		nb_iter = random.randint(a=nb_iter[0], b=nb_iter[1])

	attack = torchattacks.PGD(model=model, eps=epsilon, alpha=alpha, steps=nb_iter)
	perturbed_image = attack(rgb, label)
	return perturbed_image

def add_cw_attack(model, rgb, label, confidence, nb_iter,  box_constraints=1e-4, lr=0.01):
	if isinstance(confidence, tuple):
		confidence = random.uniform(a=confidence[0], b=confidence[1])
	if isinstance(nb_iter, tuple):
		nb_iter = random.randint(a=nb_iter[0], b=nb_iter[1])
	if isinstance(box_constraints, tuple):
		box_constraints = random.uniform(a=box_constraints[0], b=box_constraints[1])

	attack = torchattacks.CW(model=model, c=box_constraints, kappa=confidence, steps=nb_iter, lr=lr)
	perturbed_image = attack(rgb, label)
	return perturbed_image

def add_gan_attack(rgb, gan_model_weights):
	adv_GAN_model = AdvGAN_Attack('cuda', 3, 0, 1)
	gan_model_weights_absolute_path = '{}/{}'.format(os.environ['GAN_MODELS_ROOT'], gan_model_weights)
	adv_GAN_model.netG.load_state_dict(torch.load(gan_model_weights_absolute_path))

	perturbation = adv_GAN_model.netG(rgb)
	perturbed_image = torch.clamp(perturbation, -0.3, 0.3) + rgb
	perturbed_image = torch.clamp(perturbed_image, adv_GAN_model.box_min, adv_GAN_model.box_max)

	#yo_yo = perturbed_image - rgb
	#print('yo yo test: ', yo_yo)
	#print('yo yo test max: ', torch.max(yo_yo))

	return perturbed_image



