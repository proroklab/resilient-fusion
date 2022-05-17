import streamlit as st
import numpy as np
import time

import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import components.nav.ui as nav_ui
import components.data.ui as data_ui
import components.inference.ui as inf_ui

import components.data.functional as data_fn
import components.inference.functional as inf_fn
import components.perturbations.image.functional as image_perturbations_fn
import components.perturbations.lidar.functional as lidar_perturbations_fn

from leaderboard.utils.result_writer import ResultOutputProvider

def app():
	#if st.session_state.inference_step_cntr == 0: ## first perturbation has already been applied when you land on results page after selecting the settings
	#	st.session_state.steps_since_last_perturbation = 1

	st.title('Results')

	msg = None
	if st.session_state.original_image is None and st.session_state.original_lidar is None:
		msg = 'Camera and LiDAR readings not available with current settings.'
	elif st.session_state.original_image is None:
		msg = 'Camera readings not available with current settings.'
	elif st.session_state.original_lidar is None:
		msg = 'LiDAR readings not available with current settings.'

	if msg is not None and len(msg) > 0:
		st.markdown('<p style="color:red;font-size:24px;">{}</p>'.format(msg), unsafe_allow_html=True)

	image_col, lidar_col, chart_col = st.columns((0.5, 0.5, 1))
	image_col.header('RGB')
	lidar_col.header('LiDAR')
	chart_col.header('Report')

	if st.session_state.original_image is not None:
		data_ui.display_image(col=image_col, 
						image=st.session_state.original_image, 
						caption='Original'
					)
		data_ui.display_image(col=image_col, 
						image=st.session_state.perturbed_image, 
						caption='Perturbed'
					)

	if st.session_state.original_lidar is not None:
		data_ui.display_image(lidar_col, data_fn.lidar_2darray_to_rgb(st.session_state.original_lidar), caption='Original')
		data_ui.display_image(lidar_col, data_fn.lidar_2darray_to_rgb(st.session_state.perturbed_lidar), caption='Perturbed')

	st.header('Plots')
	chart_speed_col, chart_heading_col = st.columns((1, 1))

	if len(st.session_state.carla_ego_stats) > 0:
		fig_acc = px.line(st.session_state.carla_ego_stats, y='acceleration', labels={'index': 'Step', 'acceleration': 'Acceleration'}, title='Ego Acceleration over Time')
		fig_speed = px.line(st.session_state.carla_ego_stats, y=['speed', 'expected_speed'], labels={'index': 'Step', 'speed': 'Actual Speed', 'expected_speed': 'Expected Speed'}, title='Actual vs Expected Speed over Time')
		fig_heading = px.line(st.session_state.carla_ego_stats, y=['heading', 'expected_heading'], labels={'index': 'Step', 'heading': 'Actual Heading', 'expected_heading': 'Expected Heading'}, title='Actual vs Expected Heading over Time')
		
		#fig_acc = px.line(st.session_state.carla_ego_stats, x='L', y='acceleration', labels={'index': 'Step', 'acceleration': 'Acceleration'}, title='Ego Acceleration over Time')
		#fig_speed = px.line(st.session_state.carla_ego_stats, x='L', y=['speed', 'expected_speed'], labels={'index': 'Step', 'speed': 'Actual Speed', 'expected_speed': 'Expected Speed'})
		#fig_heading = px.line(st.session_state.carla_ego_stats, x='L', y=['heading', 'expected_heading'], labels={'index': 'Step', 'heading': 'Actual Heading', 'expected_heading': 'Expected Heading'})
	
		chart_speed_col.plotly_chart(fig_speed, use_container_width=True)
		chart_heading_col.plotly_chart(fig_heading, use_container_width=True)
		#chart_acc_col.plotly_chart(fig_acc, use_container_width=True)

	prev_col, _, inference_col, _, next_col = st.columns((1, 0.2, 1, 0.2, 1))
	nav_ui.display_prev_button(prev_col)
	nav_ui.display_next_button(next_col, label='Back to home')
	if st.session_state.original_image is not None or st.session_state.original_lidar is not None:
		#nav_ui.display_next_sample_button(next_col)
		inf_ui.display_inference_button(inference_col, st.session_state.num_frames_to_perturb_selected)

	if (st.session_state.inference_step_cntr == st.session_state.num_frames_to_perturb_selected) or (st.session_state.data_source_selected == 'CARLA' and st.session_state.carla_leaderboard_evaluator is not None and not st.session_state.carla_leaderboard_evaluator.manager._running):
		st.error('inf loop is done')
		st.session_state.inference_step_cntr = 0
		st.session_state.perform_inference_flag = False

	if st.session_state.data_source_selected == 'CARLA' and st.session_state.carla_leaderboard_evaluator is not None and not st.session_state.carla_leaderboard_evaluator.manager._running:
		results_text = st.session_state.carla_leaderboard_evaluator.stop_scenario(st.session_state.carla_args)
		st.session_state.carla_ego_stats.to_csv('{}/ego_stats.csv'.format(st.session_state.results_folder_path), encoding='utf-8', index=False)
		ego_lengths, ego_widths = st.session_state.carla_leaderboard_evaluator.get_ego_vehicle_size()
		st.session_state.carla_driving_metrics.aggregate(ego_lengths[0], ego_widths[0], results_text)

		st.info('Route completed!')

	if st.session_state.perform_inference_flag and st.session_state.inference_step_cntr < st.session_state.num_frames_to_perturb_selected:
		if st.session_state.data_source_selected == 'CARLA' and st.session_state.carla_leaderboard_evaluator is not None and st.session_state.carla_leaderboard_evaluator.manager._running:
			
			acc = st.session_state.carla_leaderboard_evaluator.get_ego_acceleration()[0]
			speed = st.session_state.carla_leaderboard_evaluator.get_ego_speed()[0]
			loc = st.session_state.carla_leaderboard_evaluator.get_ego_location()[0]
			heading = st.session_state.carla_leaderboard_evaluator.get_ego_heading()[0]
			is_at_traffic_light = st.session_state.carla_leaderboard_evaluator.get_is_at_traffic_light()[0]
			dist_to_closest_actor = st.session_state.carla_leaderboard_evaluator.get_dist_to_closest_actor()[0]
			st.session_state.carla_ego_stats = st.session_state.carla_ego_stats.append({'acceleration': acc, 'speed': speed, 'x': loc[0], 'y': loc[1], 'heading': heading, 'expected_heading': -999, 'expected_speed': -999, 'projected_x': -999, 'projected_y':-999, 'L': -999}, ignore_index=True)

			#st.session_state.carla_driving_metrics.compute_heading_error(heading, loc)
			st.session_state.carla_driving_metrics.compute(loc, heading, speed, is_at_traffic_light, dist_to_closest_actor)

			#inf_fn.agent_performance_clean_vs_noisy_data()
			inf_fn.perform_carla_inference_step()
			st.session_state.steps_since_last_perturbation += 1

			data_fn.get_data_from_carla()

			st.error('Steps since last perturb: {}, {}'.format(st.session_state.steps_since_last_perturbation, st.session_state.num_frames_to_skip_selected))
			if st.session_state.num_frames_to_skip_selected == st.session_state.steps_since_last_perturbation -1:
				if st.session_state.original_image is not None:
					image_perturbations_fn.apply_all_perturbations()
				if st.session_state.original_lidar is not None:
					lidar_perturbations_fn.apply_all_perturbations()
					lidar_perturbations_fn.apply_all_perturbations() ## hack to make lidar perturbations work; TODO: fix the bug causing this issue

				st.session_state.steps_since_last_perturbation = 0

		st.info('inf: {}, {}/{}, {}'.format(st.session_state.perform_inference_flag, st.session_state.inference_step_cntr, st.session_state.num_frames_to_perturb_selected, st.session_state.carla_leaderboard_evaluator.manager._running))
		time.sleep(1)
		st.session_state.inference_step_cntr += 1
		st.experimental_rerun()