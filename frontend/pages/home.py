import streamlit as st
import numpy as np

import pandas as pd
import pydeck

import components.nav.ui as nav_ui
import components.data.ui as data_ui
import components.data.functional as data_fn

def app():
	st.title('Home')

	msg = None
	if st.session_state.original_image is None and st.session_state.original_lidar is None:
		msg = 'Camera and LiDAR readings not available with current settings.'
	elif st.session_state.original_image is None:
		msg = 'Camera readings not available with current settings.'
	elif st.session_state.original_lidar is None:
		msg = 'LiDAR readings not available with current settings.'

	if msg is not None and len(msg) > 0:
		st.markdown('<p style="color:red;font-size:24px;">{}</p>'.format(msg), unsafe_allow_html=True)

	st.subheader('Data sample to modify')

	image_col, _, lidar_col = st.columns((1, 0.5, 1))
	if st.session_state.original_image is not None:
		data_ui.display_image(image_col, st.session_state.original_image)
	if st.session_state.original_lidar is not None:
		data_ui.display_image(lidar_col, data_fn.lidar_2darray_to_rgb(st.session_state.original_lidar), caption='Original')
	#	data_ui.display_point_cloud(lidar_col, st.session_state.original_lidar)

	_, col = st.columns((2, 1))
	nav_ui.display_next_button(col, 'Modify Sample')
	if st.session_state.original_image is not None or st.session_state.original_lidar is not None:
		#nav_ui.display_modify_sample_button(col)
		nav_ui.display_next_sample_button(col)





	#st.text('Data index: {} {}'.format(st.session_state.dataloader_index, st.session_state.data['index']))

	"""
	_, image_col, _, lidar_col, _ = st.columns((0.5, 1, 0.5, 1, 0.5))
	
	image_col.header('RGB')
	if st.session_state.original_image is not None:
		image_col.image(st.session_state.original_image)
	#display_image(col=image_col, 
	#				image=st.session_state.original_image,
	#			)

	lidar_col.header('Lidar')
	#display_image(col=lidar_col, 
	#				image=lidar_utils.lidar_2darray_to_rgb(st.session_state.original_lidar),
	#			)
	
	if st.session_state.original_lidar is not None:
		lidar_ui.display_point_cloud(lidar_col, st.session_state.original_lidar)
		'''
		df = pd.DataFrame(data=st.session_state.original_lidar, columns=["x", "y", "z"])
		point_cloud_layer = pydeck.Layer(
			"PointCloudLayer",
			data=df,
			get_position=["x", "y", "z"],
			get_normal=[0, 0, 15],
			auto_highlight=True,
			point_size=2,
		)
		r = pydeck.Deck(point_cloud_layer, width=10, height=10, map_provider=None)
		#lidar_col.write(r)
		lidar_col.pydeck_chart(r, use_container_width=False)
		'''

	_, col = st.columns((2, 1))
	nav_ui.display_modify_sample_button(col)
	nav_ui.display_next_sample_button(col)
	"""