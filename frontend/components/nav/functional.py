import streamlit as st

import math

from pages import image_modification, lidar_modification, home, results
import components.data.functional as data_fn
import components.inference.functional as inf_fn

def get_prev_page(current_page):
	PREV_PAGES = {
			'Image modification': ['Home', home.app],
			'LiDAR modification': ['Image modification', image_modification.app],
			'Results': ['LiDAR modification', lidar_modification.app],
		}

	st.session_state.page_selected = {
				'page_name': PREV_PAGES[current_page][0], 
				'function': PREV_PAGES[current_page][1],
			}

def get_next_page(current_page):
	NEXT_PAGES = {
			'Home': ['Image modification', image_modification.app],
			'Image modification': ['LiDAR modification', lidar_modification.app],
			'LiDAR modification': ['Results', results.app],
			'Results': ['Home', home.app],
		}

	#print('HELLOOOOOO: ', NEXT_PAGES[current_page])

	st.session_state.page_selected = {
				'page_name': NEXT_PAGES[current_page][0], 
				'function': NEXT_PAGES[current_page][1],
			}

def get_next_sample(current_page):

	#st.write('Next sample selected!')
	#if st.session_state.data_source_selected == 'CARLA':
	#	data_fn.get_data_from_carla()

	if st.session_state.data_source_selected == 'CARLA':
		inf_fn.perform_carla_inference_step()
		data_fn.get_data_from_carla()

	'''
	if current_page == 'Results':
		st.session_state.page_selected = {
				'page_name': 'Home', 
				'function': home.app
			}
	'''