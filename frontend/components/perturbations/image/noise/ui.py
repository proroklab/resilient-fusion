import streamlit as st

import components.perturbations.image.noise.functional as noise_fn

def display_controls(col):
	with col.expander('Noise'):

		noise_type = st.radio(label='Type',
								options=('Gaussian', 'Salt', 'Pepper', 'S&P'),
								key='image_noise_type',
						)

		noise_form = st.form('image_noise_form')
		if noise_type.lower() in ['salt', 'pepper', 's&p']:
			noise_form.slider('Amount: ', 
								min_value=0.0, 
								max_value=1.0, 
								value=(0.0, 0.2) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.0, 
								step=0.01, 
								key='image_sp_amount',
						)
		else:
			noise_form.slider('Variance: ', 
								min_value=0.0, 
								max_value=1.0, 
								value=(0.0, 0.2) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.0, 
								step=0.01, 
								on_change=None, 
								key='image_gaussian_variance',
							)

		noise_form.form_submit_button('Add noise', 
										on_click=noise_fn.update_perturbation_settings,
								)