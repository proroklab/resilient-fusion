import streamlit as st

import os
import glob

import components.perturbations.image.adversarial.functional as adversarial_fn

def display_controls(col):
	with col.expander('Adversarial'):

		attack_type = st.radio(label='Type',
								options=('FGSM', 'PGD', 'Carlini-Wagner', 'GAN'),
								key='image_attack_type',
						)

		attack_form = st.form('image_attack_form')
		if attack_type.lower() == 'fgsm':
			attack_form.slider('Epsilon: ', 
								min_value=0.0, 
								max_value=1.0, 
								value=(0.0, 0.2) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.0, 
								step=0.01,
								key='image_fgsm_epsilon',
							)

		elif attack_type.lower() == 'pgd':
			attack_form.slider('Epsilon: ', 
								min_value=0.0, 
								max_value=1.0, 
								value=(0.0, 0.2) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.0, 
								step=0.01,
								key='image_pgd_epsilon',
							)
			attack_form.slider("Alpha: ", 
								min_value=0.0, 
								max_value=1.0, 
								value=(0.0, 0.2) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.0, 
								step=0.01,
								key='image_pgd_alpha',
							)
			attack_form.slider("Iterations: ", 
								min_value=1, 
								max_value=200, 
								value=(0, 20) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0, 
								step=9,
								key='image_pgd_iters',
							)

		elif attack_type.lower() == 'carlini-wagner':
			attack_form.slider('Epsilon: ', 
								min_value=0.0, 
								max_value=1.0, 
								value=(0.0, 0.2) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.0, 
								step=0.01,
								key='image_cw_epsilon',
							)
			'''
			attack_form.slider('Binary Search Steps: ', 
								min_value=10, 
								max_value=200, 
								value=(1, 30) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 1, 
								step=10, 
								key='image_cw_binary_search_steps',
							)
			'''
			
			attack_form.slider('Box constraints: ', 
								min_value=0.0, 
								max_value=1.0, 
								value=(0.0, 0.2) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.0, 
								step=0.01, 
								key='image_cw_box_constraints',
							)

			attack_form.slider("Iterations: ", 
								min_value=0, 
								max_value=10000, 
								value=(0, 200) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0, 
								step=100,
								key='image_cw_iters',
							)

		elif attack_type.lower() == 'gan':
			GAN_ROOT = os.environ['GAN_MODELS_ROOT']
			available_paths = glob.glob('{}/*.pth'.format(GAN_ROOT))
			print('Av paths before: ', available_paths)
			available_paths = [p.split('/')[-1] for p in available_paths]

			attack_form.selectbox('GAN model: ', options=available_paths, key='image_gan_model_weights')
			#st.session_state['image_gan_model_weights'] = '{}/{}'.format(GAN_ROOT, st.session_state['image_gan_model_weights'])
			#print('Sel path: ', st.session_state['image_gan_model_weights'])
			#attack_form.file_uploader('GAN trained model: ',
			#					key='image_gan_model_weights',
			#				)
		

		attack_form.form_submit_button('Add attack',
										on_click=adversarial_fn.update_perturbation_settings,
									)