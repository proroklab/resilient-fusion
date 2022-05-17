import streamlit as st

import components.perturbations.image.occlusion.functional as occlusion_fn


def display_controls(col):
	with col.expander('Occlusion'):
		occ_form = st.form('image_occlusion_form')
		occ_form.slider('x position: ', 
								min_value=0.0, 
								max_value=1.0, 
								value=(0.4, 0.6) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.4, 
								step=0.01,
								key='image_occlusion_x'
							)

		occ_form.slider('y position: ', 
								min_value=0.0, 
								max_value=1.0, 
								value=(0.4, 0.6) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.4, 
								step=0.01,
								key='image_occlusion_y'
							)

		occ_form.slider('Width: ', 
								min_value=0.0, 
								max_value=1.0, 
								value=(0.0, 0.2) if st.session_state.perturbation_settings_mode_selected == 'Interval' else 0.0, 
								step=0.01,
								key='image_occlusion_width'
							)

		#occ_form.selectbox('Colour: ',
		#						options=('Black', 'Blue', 'Red', 'Green', 'White'),
		#						key='image_occlusion_colour'
		#					)

		occ_form.form_submit_button('Add occlusion', 
										on_click=occlusion_fn.update_perturbation_settings,
								)