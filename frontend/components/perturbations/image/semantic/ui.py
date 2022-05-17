import streamlit as st

import components.perturbations.image.semantic.functional as semantic_fn


def display_controls(col):
	with col.expander('Hallucination'):
		hallucination_form = st.form('image_hallucination_form')
		hallucination_form.text('All measurements are in percentage')
		hallucination_form.slider('Centre x position: ', 
								min_value=0.0, 
								max_value=1.0, 
								value=(0.4, 0.6) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.4, 
								step=0.01,
								key='image_hallucination_x'
							)

		hallucination_form.slider('Centre y position: ', 
								min_value=0.0, 
								max_value=1.0, 
								value=(0.6, 0.8) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.6, 
								step=0.01,
								key='image_hallucination_y'
							)

		hallucination_form.slider('Size: ', 
								min_value=0.0, 
								max_value=1.0,
								value=(0.0, 0.2) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.0,
								step=0.01,
								key='image_hallucination_size'
							)

		hallucination_form.form_submit_button('Add hallucination', 
										on_click=semantic_fn.update_perturbation_settings,
								)