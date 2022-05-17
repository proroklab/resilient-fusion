import streamlit as st

import components.perturbations.lidar.semantic.functional as semantic_fn

def display_controls(col):
	with col.expander('Hallucination'):

		hallucination_form = st.form('lidar_semantic_form')
		hallucination_form.slider('Distance to ego:',
								min_value=0.0, 
								max_value=1.0, 
								value=(0.0, 0.2) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.0, 
								step=0.01, 
								key='lidar_hallucination_distance'
						)

		hallucination_form.slider('Position:',
								min_value=0.0, 
								max_value=1.0, 
								value=(0.0, 0.2) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.0, 
								step=0.01, 
								key='lidar_hallucination_position'
						)

		hallucination_form.slider('Width: ', 
								min_value=0.0, 
								max_value=1.0, 
								value=(0.0, 0.2) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.0, 
								step=0.01, 
								key='lidar_hallucination_width'
						)
		hallucination_form.slider('Height: ', 
								min_value=0.0, 
								max_value=1.0, 
								value=(0.0, 0.2) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.0, 
								step=0.01, 
								key='lidar_hallucination_height'
						)

		hallucination_form.form_submit_button('Add hallucination', 
										on_click=semantic_fn.update_perturbation_settings,
									)