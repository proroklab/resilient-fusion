import streamlit as st
import numpy as np

import components.nav.ui as nav_ui
import components.data.ui as data_ui
import components.perturbations.image.noise.ui as noise_ui
import components.perturbations.image.blur.ui as blur_ui
import components.perturbations.image.adversarial.ui as adversarial_ui
import components.perturbations.image.occlusion.ui as occlusion_ui
import components.perturbations.image.semantic.ui as semantic_ui
import components.perturbations.image.functional as perturbations_fn

def app():
	st.title('Image Modification')

	if st.session_state.original_image is None:
		#st.text('Camera readings not available with current settings.')
		st.markdown(f'<p style="color:red;font-size:24px;">Camera readings not available with current settings.</p>', unsafe_allow_html=True)

	settings_col, _, mods_col, _, images_col = st.columns((1, 0.2, 1, 0.2, 1))
	settings_col.header('Settings')

	if st.session_state.original_image is not None:
		adversarial_ui.display_controls(col=settings_col)
		blur_ui.display_controls(col=settings_col)
		noise_ui.display_controls(col=settings_col)
		occlusion_ui.display_controls(col=settings_col)
		semantic_ui.display_controls(col=settings_col)
	

	mods_col.header('Modifications selected')
	mods_col.write(st.session_state.image_perturbation_settings)
	if st.session_state.original_image is None:
		st.session_state.image_perturbation_settings = {}
	if len(st.session_state.image_perturbation_settings.keys()) > 0:
		mods_col.button('Reset all', on_click=perturbations_fn.reset_all_perturbations)
		mods_col.button('Apply all', on_click=perturbations_fn.apply_all_perturbations)

	images_col.header('Images')
	if st.session_state.original_image is not None:
		data_ui.display_image(col=images_col, image=st.session_state.original_image, caption='Original')
		data_ui.display_image(col=images_col, image=st.session_state.perturbed_image, caption='Perturbed')

	'''
	if st.button('Testing'):
		import torch
		import torchattacks
		import components.inference.functional as inf_fn

		input_image = st.session_state.original_image
		le = st.session_state.carla_leaderboard_evaluator
		model = le.agent_instance.get_image_encoder_model()

		rgb = inf_fn.prepare_image_for_inference(input_image)
		image_features = le.agent_instance.get_image_features(model, rgb)
		rgb /= 255.
		label = torch.argmax(image_features)
		label = torch.LongTensor([label])

		attack = torchattacks.OnePixel(model, pixels=1000)
		perturbed_im = attack(rgb, label)

		perturbed_im = perturbed_im.squeeze(0).permute(1, 2, 0)
		perturbed_im *= 255.0
		perturbed_im = perturbed_im.detach().cpu().numpy().astype(np.uint8)

		### only added for testing purposes
		pert_im = inf_fn.prepare_image_for_inference(perturbed_im)
		perturbed_image_features = le.agent_instance.get_image_features(model, pert_im)
		perturbed_label = torch.argmax(perturbed_image_features)
		perturbed_label = torch.LongTensor([perturbed_label])
		print('Testing testing adv attack: ', label, perturbed_label)

		st.image(perturbed_im, caption='yo yo')
	'''



	prev_col, _, next_col = st.columns((0.5, 1.8, 0.5))
	nav_ui.display_prev_button(prev_col)
	nav_ui.display_next_button(next_col)