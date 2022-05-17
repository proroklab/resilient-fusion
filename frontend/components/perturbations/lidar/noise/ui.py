import streamlit as st

import components.perturbations.lidar.noise.functional as noise_fn

def display_controls(col):
	with col.expander('Noise'):

		noise_form = st.form('lidar_noise_form')
		noise_form.slider('Amount: ', 
								min_value=0.0, 
								max_value=1.0, 
								value=(0.0, 0.2) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.0, 
								step=0.01, 
								key='lidar_noise_amount'
						)
		noise_form.slider('Percentage of points to perturb: ', 
								min_value=0.0, 
								max_value=1.0, 
								value=(0.0, 0.2) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.0, 
								step=0.01, 
								key='lidar_perturb_percent'
						)

		noise_form.form_submit_button('Add noise', 
										on_click=noise_fn.update_perturbation_settings,
									)