import streamlit as st

import components.perturbations.image.blur.functional as blur_fn


def display_controls(col):
	with col.expander('Motion Blur'):
		blur_form = st.form('image_blur_form')
		blur_form.slider('Amount: ', 
								min_value=0.0, 
								max_value=1.0, 
								value=(0.0, 0.2) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.0, 
								step=0.01,
								key='image_blur_amount'
							)
		blur_form.form_submit_button('Add blur', 
										on_click=blur_fn.update_perturbation_settings,
								)