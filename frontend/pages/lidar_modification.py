import streamlit as st
import numpy as np

import components.nav.ui as nav_ui
import components.data.ui as data_ui
import components.perturbations.lidar.noise.ui as noise_ui
import components.perturbations.lidar.adversarial.ui as adversarial_ui
import components.perturbations.lidar.semantic.ui as semantic_ui

import components.data.functional as data_fn
import components.perturbations.lidar.functional as perturbations_fn

def app():
	st.title('LiDAR Modification')

	if st.session_state.original_lidar is None:
		#st.text('LiDAR readings not available with current settings.')
		st.markdown(f'<p style="color:red;font-size:24px;">LiDAR readings not available with current settings.</p>', unsafe_allow_html=True)

	settings_col, _, mods_col, _, lidars_col = st.columns((1, 0.2, 1, 0.2, 1))

	settings_col.header('Settings')
	if st.session_state.original_lidar is not None:
		noise_ui.display_controls(settings_col)
		adversarial_ui.display_controls(settings_col)
		semantic_ui.display_controls(settings_col)

	mods_col.header('Modifications selected')
	mods_col.write(st.session_state.lidar_perturbation_settings)
	if st.session_state.original_lidar is None:
		st.session_state.lidar_perturbation_settings = {}
	if len(st.session_state.lidar_perturbation_settings.keys()) > 0:
		mods_col.button('Reset all', on_click=perturbations_fn.reset_all_perturbations)
		mods_col.button('Apply all', on_click=perturbations_fn.apply_all_perturbations)

	lidars_col.header('LiDAR')
	if st.session_state.original_lidar is not None:
		#data_ui.display_point_cloud(lidars_col, st.session_state.original_lidar, caption='Original')
		#data_ui.display_point_cloud(lidars_col, st.session_state.perturbed_lidar, caption='Perturbed')
		data_ui.display_image(lidars_col, data_fn.lidar_2darray_to_rgb(st.session_state.original_lidar), caption='Original')
		data_ui.display_image(lidars_col, data_fn.lidar_2darray_to_rgb(st.session_state.perturbed_lidar), caption='Perturbed')
		#data_ui.display_image(lidars_col, data_fn.lidar_2darray_to_rgb(st.session_state.original_lidar_im), caption='Original')
		#data_ui.display_image(lidars_col, data_fn.lidar_2darray_to_rgb(st.session_state.perturbed_lidar_im), caption='Perturbed')

	prev_col, _, next_col = st.columns((0.5, 1.8, 0.5))
	nav_ui.display_prev_button(prev_col)
	nav_ui.display_next_button(next_col)