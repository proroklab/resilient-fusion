import streamlit as st

import components.perturbations.lidar.adversarial.functional as adversarial_fn

def display_controls(col):
	with col.expander('Adversarial'):

		attack_type = st.radio(label='Type',
								options=('FGSM', 'PGD', 'Carlini-Wagner'),
								key='lidar_attack_type',
						)

		attack_form = st.form('lidar_attack_form')
		if attack_type.lower() == 'fgsm':
			attack_form.slider('Epsilon: ', 
								min_value=0.0, 
								max_value=1.0, 
								value=(0.0, 0.2) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.0, 
								step=0.01,
								key='lidar_fgsm_epsilon',
							)

		elif attack_type.lower() == 'pgd':
			attack_form.slider('Epsilon: ', 
								min_value=0.0, 
								max_value=1.0, 
								value=(0.0, 0.2) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.0, 
								step=0.01,
								key='lidar_pgd_epsilon',
							)
			attack_form.slider("Alpha: ", 
								min_value=0.0, 
								max_value=1.0, 
								value=(0.0, 0.2) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.0, 
								step=0.01,
								key='lidar_pgd_alpha',
							)
			attack_form.slider("Iterations: ", 
								min_value=1, 
								max_value=200, 
								value=(0, 20) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0, 
								step=9,
								key='lidar_pgd_iters',
							)

		elif attack_type.lower() == 'carlini-wagner':
			attack_form.slider('Epsilon: ', 
								min_value=0.0, 
								max_value=1.0, 
								value=(0.0, 0.2) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.0, 
								step=0.01,
								key='lidar_cw_epsilon',
							)
			
			attack_form.slider('Box constraints: ', 
								min_value=0.0, 
								max_value=1.0, 
								value=(0.0, 0.2) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.0, 
								step=0.01, 
								key='lidar_cw_box_constraints',
							)

			attack_form.slider("Iterations: ", 
								min_value=0, 
								max_value=10000, 
								value=(0, 200) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0, 
								step=100,
								key='lidar_cw_iters',
							)
		
		attack_form.form_submit_button('Add attack',
										on_click=adversarial_fn.update_perturbation_settings,
									)