import streamlit as st

import components.perturbations.image.adversarial.functional as adv_fn
import components.perturbations.image.blur.functional as blur_fn
import components.perturbations.image.noise.functional as noise_fn
import components.perturbations.image.occlusion.functional as occlusion_fn
import components.perturbations.image.semantic.functional as semantic_fn

def apply_all_perturbations():
	perturbed_image = st.session_state.original_image

	occlusion_settings = st.session_state['image_perturbation_settings'].get('occlusion', None)
	if occlusion_settings is not None:
		perturbed_image = occlusion_fn.apply_occlusion(input_image=perturbed_image, settings=occlusion_settings)

	hallucination_settings = st.session_state['image_perturbation_settings'].get('hallucination', None)
	if hallucination_settings is not None:
		perturbed_image = semantic_fn.apply_hallucination(input_image=perturbed_image, settings=hallucination_settings)

	blur_settings = st.session_state['image_perturbation_settings'].get('blur', None)
	if blur_settings is not None:
		perturbed_image = blur_fn.apply_blur(input_image=perturbed_image, settings=blur_settings)
	
	noise_settings = st.session_state['image_perturbation_settings'].get('noise', None)
	if noise_settings is not None:
		perturbed_image = noise_fn.apply_noise(input_image=perturbed_image, settings=noise_settings)

	adversarial_settings = st.session_state['image_perturbation_settings'].get('adversarial', None)
	if adversarial_settings is not None:
		perturbed_image = adv_fn.apply_adversarial_attack(input_image=perturbed_image, settings=adversarial_settings)
	
	st.session_state.perturbed_image = perturbed_image

def reset_all_perturbations():
	st.session_state.image_perturbation_settings = {}
	st.session_state.perturbed_image = st.session_state.original_image