import streamlit as st

import bisect
import os
import math
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from pathlib import Path
from tabulate import tabulate
import pickle

import carla
from srunner.scenariomanager.traffic_events import TrafficEventType
from agents.navigation.local_planner import RoadOption

from collections import deque

LANETYPE_TO_SPEED = {
	RoadOption.RIGHT: 3.5,
	RoadOption.LEFT: 3.5,
	RoadOption.CHANGELANERIGHT: 4,
	RoadOption.CHANGELANELEFT: 4,
	RoadOption.LANEFOLLOW: 6.5,
	RoadOption.STRAIGHT: 6.5,
}

def calculate_speed(velocity):
	if not isinstance(velocity, np.ndarray):
		velocity = np.array(velocity)
	return np.sqrt(np.sum(velocity ** 2))

def calculate_distance(pt1, pt2):
	if not isinstance(pt1, np.ndarray):
		pt1 = np.array(pt1)

	if not isinstance(pt2, np.ndarray):
		pt2 = np.array(pt2)

	return np.sqrt(np.sum((pt1-pt2) ** 2))

class DrivingMetrics:
	def __init__(self, baseline_traj, potential_static_collisions, potential_vehicle_collisions, potential_pedestrian_collisions):
		self.baseline_traj_wps, self.baseline_traj_lanetypes = self._extract_baseline_traj_wps_and_lanetypes(baseline_traj)
		self.lanetypes_in_route = set(self.baseline_traj_lanetypes)
		print('LANETYPES: ', self.lanetypes_in_route)

		#for i, el in enumerate(baseline_traj):
		#	print(i, el[1])

		self.section_end_indices = self._determine_route_section_ends()
		self.num_of_sections = len(self.section_end_indices)-1

		print('Section indices: ', self. section_end_indices)
		print('Num sections: ', self.num_of_sections)

		num_prev_speed = 3
		self.prev_speed = deque([0]*num_prev_speed, maxlen=num_prev_speed)
		self.last_projected_loc = self.baseline_traj_wps[0]

		self.last_L_in_section = [0] * self.num_of_sections
		self.heading_error_per_section = [0] * self.num_of_sections
		self.speed_error_per_section = [0] * self.num_of_sections
		self.path_error_per_section = [0] * self.num_of_sections

		self.total_L_per_section = self._compute_L_per_route_section()
		self.avg_expected_heading_per_section = self._compute_average_expected_heading_per_section()
		self.avg_expected_speed_per_section = self._compute_average_expected_speed_per_section()
		#self.avg_expected_speed_per_section = [0] * self.num_of_sections
		#self.speed_cnts = [0] * self.num_of_sections

		self.potential_static_collisions = potential_static_collisions
		self.potential_vehicle_collisions = potential_vehicle_collisions
		self.potential_pedestrian_collisions = potential_pedestrian_collisions
		#for c in self.potential_static_collisions:
		#	print('Static: ', c._reference_waypoint)

		#self.static_collisions_locations = [[c['x'], c['y']]for c in self.potential_static_collisions]
		#print('Static col locs: ', self.static_collisions_locations)

		#print('Avg heading per section: ', self.avg_expected_heading_per_section)
		#print('Avg speed per section: ', self.avg_expected_speed_per_section)


		'''
		

		self.heading_actual = []
		self.heading_expected = []
		self.speed_actual = []
		self.speed_expected = []
		
		self.heading_error_per_section = [0] * self.total_path_sections
		self.speed_error_per_section = [0] * self.total_sections
		self.path_error_per_section = [0] * self.total_path_sections

		self.L = 0
		self.last_L = [0] * self.total_path_sections
		self.last_projected_loc = self.baseline_traj[0]

		self.step_cntr = 0
		'''

	def _extract_baseline_traj_wps_and_lanetypes(self, baseline_traj):
		dense_traj = []
		lanetypes = []
		for i, el in enumerate(baseline_traj):
			#print('EL: ', el)
			l = el[0].location
			dense_traj.append([l.x, l.y])
			lanetypes.append(el[1])
		return np.array(dense_traj), np.array(lanetypes)

	def _determine_route_section_ends(self):
		section_end_indices = [0]
		for idx in range(1, len(self.baseline_traj_lanetypes)):
			## if the lanetype is different from the previous one, track its corresponding index as the end point of the section
			if self.baseline_traj_lanetypes[idx] != self.baseline_traj_lanetypes[idx-1]:
				if len(section_end_indices) > 0 and section_end_indices[-1] != 0 and idx - section_end_indices[-1] == 1:
					section_end_indices[-1] = idx
				else:
					section_end_indices.append(idx)
		return np.array(section_end_indices)

	def _project_to_route(self, loc_xy):
		dist_to_route_wps = np.sum((self.baseline_traj_wps - loc_xy)**2, axis=1)
		#print('Dists: ', dist_to_route_wps)
		idx_closest_wp = np.argmin(dist_to_route_wps)
		return self.baseline_traj_wps[idx_closest_wp], idx_closest_wp, dist_to_route_wps[idx_closest_wp]

	def _compute_L(self, start_idx, end_idx):
		left_pts = self.baseline_traj_wps[start_idx: end_idx]
		right_pts = self.baseline_traj_wps[start_idx+1: end_idx+1]

		dist = np.linalg.norm(left_pts - right_pts)
		return dist

	def _get_current_route_section(self, projected_loc_idx):
		## find where in the section_end_indices the idx of the projected location lies, this allows to determine the section of the route that the location falls within
		projected_loc_in_section_idx = bisect.bisect(self.section_end_indices, projected_loc_idx)
		section_number = min(max(0, projected_loc_in_section_idx-1), self.num_of_sections-1) ## the element before and after the position of the projected_loc_idx are the lower and higher ends of the respective section 
		return section_number, self.section_end_indices[section_number], self.section_end_indices[section_number+1]


	def _compute_expected_heading(self, projected_loc_idx):
		projected_loc_idx_next = min(projected_loc_idx+2, len(self.baseline_traj_wps)-1)
		direction_vector = self.baseline_traj_wps[projected_loc_idx_next] - self.baseline_traj_wps[projected_loc_idx]
		return np.arccos(direction_vector[0] / (0.000001 + np.linalg.norm(direction_vector)))

	def _compute_expected_speed(self, projected_loc_idx, is_at_traffic_light, dist_to_closest_actor):
		#if projected_loc_idx == 0 or is_at_traffic_light:
		if is_at_traffic_light:
			return 0

		actor_dist = 999
		if dist_to_closest_actor is not None:
			for d in dist_to_closest_actor:
				pl, pli, pld = self._project_to_route((d[1], d[2]))
				print('PLD: ', projected_loc_idx, pld, d[0])
				if pli > projected_loc_idx and pld <= 4 and d[0] <= 11:
					return max(0, (self.prev_speed[-1] - 1))
					#actor_dist = d[0]
					#break

		#if actor_dist != 999:

				

		#print('Dist: ', dist_to_closest_actor)
		#if dist_to_closest_actor <= 10.5:
		#	return max(0, (self.prev_speed[-1] - 0.5))

		speeds = [LANETYPE_TO_SPEED[self.baseline_traj_lanetypes[projected_loc_idx]]]
		speeds += self.prev_speed

		next_speeds = [LANETYPE_TO_SPEED[self.baseline_traj_lanetypes[min(projected_loc_idx+i, len(self.baseline_traj_lanetypes)-1)]] for i in range(4)]
		next_speeds = set(next_speeds)
		speeds += list(next_speeds)


		'''
		next_speeds = []
		for i in range(2):
			if projected_loc_idx+i < len(self.baseline_traj_lanetypes):
				next_speeds.append(LANETYPE_TO_SPEED[self.baseline_traj_lanetypes[projected_loc_idx+i]])

		cnt_max_speed = 0
		for s in next_speeds:
			if s == 7 and cnt_max_speed == 0:
				speeds.append(s)
				cnt_max_speed += 1
			elif s != 7:
				speeds.append(s)
		'''

		#print('Speeds: ', speeds)

		return sum(speeds)/len(speeds)

	def _compute_L_per_route_section(self):
		L_per_section = []

		for section_number in range(self.num_of_sections):
			section_start_idx = self.section_end_indices[section_number]
			section_end_idx = self.section_end_indices[section_number+1]

			L_per_section.append(self._compute_L(section_start_idx, section_end_idx))

		return L_per_section

	def _compute_average_expected_heading_per_section(self):
		avg_heading_per_section = []

		for section_number in range(self.num_of_sections):
			section_start_idx = self.section_end_indices[section_number]
			section_end_idx = self.section_end_indices[section_number+1]
			n = 0
			heading = 0

			for idx in range(section_start_idx, section_end_idx):
				heading += self._compute_expected_heading(idx)
				n += 1

			avg_heading_per_section.append(heading/n)

		return avg_heading_per_section

	
	def _compute_average_expected_speed_per_section(self):
		avg_speed_per_section = []

		for section_number in range(self.num_of_sections):
			section_start_idx = self.section_end_indices[section_number]
			section_end_idx = self.section_end_indices[section_number+1]
			n = 0
			speed = 0 

			for idx in range(section_start_idx, section_end_idx):
				speed += self._compute_expected_speed(idx, is_at_traffic_light=False, dist_to_closest_actor=None)
				n += 1

			avg_speed_per_section.append(speed/n)

		return avg_speed_per_section
	

	def _compute_current_heading_error(self, expected_heading, ego_heading, current_L, last_L):
		heading_error = ego_heading - expected_heading
		return (current_L - last_L) * abs(heading_error)

	def _compute_current_speed_error(self, expected_speed, ego_speed, current_L, last_L):
		speed_error = ego_speed - expected_speed
		print('Curr speed error: ', current_L, last_L, speed_error, ego_speed, expected_speed)
		return (current_L - last_L) * abs(speed_error)

	#def _compute_current_path_error(self, dist_to_projected_loc, current_projected_loc, previous_projected_loc):
	#	dist_to_previous_projected_loc = np.linalg.norm(current_projected_loc - previous_projected_loc)
	#	return dist_to_previous_projected_loc * dist_to_projected_loc

	def _compute_current_path_error(self, dist_to_projected_loc, current_L, last_L):
		print('Curr path error: ', current_L, last_L, current_L-last_L, dist_to_projected_loc)
		return (current_L - last_L) * dist_to_projected_loc

	def _aggregate_heading_error(self):
		normalized_heading_error_per_section = []
		for idx in range(self.num_of_sections):
			normalized_error = self.heading_error_per_section[idx] / (self.total_L_per_section[idx] * self.avg_expected_heading_per_section[idx])
			if normalized_error != 0:
				normalized_heading_error_per_section.append(math.exp(-normalized_error))
	
		return np.sum(normalized_heading_error_per_section)/self.num_of_sections, np.sum(normalized_heading_error_per_section)/len(normalized_heading_error_per_section)

	def _aggregate_speed_error(self):
		normalized_speed_error_per_section = []

		for idx in range(self.num_of_sections):
			#normalized_error = self.speed_error_per_section[idx] / (self.total_L_per_section[idx] * (self.avg_expected_speed_per_section[idx]/self.speed_cnts[idx]))
			normalized_error = self.speed_error_per_section[idx] / (self.total_L_per_section[idx] * self.avg_expected_speed_per_section[idx])
			print(idx, self.speed_error_per_section[idx], normalized_error)
			if normalized_error != 0:
				normalized_speed_error_per_section.append(math.exp(-normalized_error))
		print('Norm speeds: ', normalized_speed_error_per_section)

		return np.sum(normalized_speed_error_per_section)/self.num_of_sections, np.sum(normalized_speed_error_per_section)/len(normalized_speed_error_per_section)

	def _aggregate_path_error(self, ego_vehicle_length):
		normalized_path_error_per_section = []

		for idx in range(self.num_of_sections):
			normalized_error = self.path_error_per_section[idx] / (self.total_L_per_section[idx] * ego_vehicle_length)
			if normalized_error != 0:
				normalized_path_error_per_section.append(math.exp(-normalized_error))

		#print('path error per section: ', normalized_path_error_per_section)

		return np.sum(normalized_path_error_per_section)/self.num_of_sections, np.sum(normalized_path_error_per_section)/len(normalized_path_error_per_section)
		#return np.sum(normalized_path_error_per_section)/len(normalized_path_error_per_section)


	def compute(self, ego_location, ego_heading, ego_speed, is_at_traffic_light, dist_to_closest_actor):
		projected_loc, projected_loc_idx, dist_to_projected_loc = self._project_to_route(np.array([ego_location[0], ego_location[1]]))
		#print('PL: ', projected_loc)
		st.session_state.carla_ego_stats.loc[st.session_state.carla_ego_stats.index[-1], 'projected_x'] = projected_loc[0]
		st.session_state.carla_ego_stats.loc[st.session_state.carla_ego_stats.index[-1], 'projected_y'] = projected_loc[1]

		section_number, section_start_idx, section_end_idx = self._get_current_route_section(projected_loc_idx)
		#print('Section: ', section_number, section_start_idx)
		current_L = self._compute_L(section_start_idx, projected_loc_idx)
		st.session_state.carla_ego_stats.loc[st.session_state.carla_ego_stats.index[-1], 'L'] = current_L
		#print('Current L: ', current_L, self.last_L)

		expected_heading = self._compute_expected_heading(projected_loc_idx)
		st.session_state.carla_ego_stats.loc[st.session_state.carla_ego_stats.index[-1], 'expected_heading'] = expected_heading

		expected_speed = self._compute_expected_speed(projected_loc_idx, is_at_traffic_light, dist_to_closest_actor)
		self.prev_speed.append(expected_speed)
		#self.avg_expected_speed_per_section[section_number] += expected_speed
		#self.speed_cnts[section_number] += 1
		st.session_state.carla_ego_stats.loc[st.session_state.carla_ego_stats.index[-1], 'expected_speed'] = expected_speed

		self.heading_error_per_section[section_number] += self._compute_current_heading_error(expected_heading, ego_heading, current_L, self.last_L_in_section[section_number])
		print('HE: ', self.heading_error_per_section)

		self.speed_error_per_section[section_number] += self._compute_current_speed_error(expected_speed, ego_speed, current_L, self.last_L_in_section[section_number])
		print('SE: ', self.speed_error_per_section)

		#self.path_error_per_section[section_number] += self._compute_current_path_error(dist_to_projected_loc, projected_loc, self.last_projected_loc)
		self.path_error_per_section[section_number] += self._compute_current_path_error(dist_to_projected_loc, current_L, self.last_L_in_section[section_number])
		print('PE: ', self.path_error_per_section)

		self.last_projected_loc = projected_loc
		self.last_L_in_section[section_number] = current_L

	def _compute_vehicle_collision_error(self, cnt_potential_vehicle_collisions, vehicle_collisions, ego_vehicle_width):
		if cnt_potential_vehicle_collisions == 0:
			return 1

		d0 = ego_vehicle_width/2
		total_error = 0

		for col in vehicle_collisions:
			obstacle_loc = np.array([col['x'], col['y']])
			ego_loc = np.array(col['ego_location'][0][:2])
			obstacle_velocity = np.array(col['other_velocity'])
			ego_velocity = np.array(col['ego_velocity'][0])
			obstacle_speed = calculate_speed(obstacle_velocity)
			ego_speed = calculate_speed(ego_velocity)

			d1 = max(abs(obstacle_loc - ego_loc)) ## TODO: using chebyshev distance here, check if it is the correct interpretation
			alpha = np.arccos((sum(ego_velocity[:2] * obstacle_velocity[:2]))/(ego_speed * obstacle_speed))
			gamma = (ego_speed * np.cos(alpha)) - obstacle_speed

			total_error += (d1 / d0) * math.exp(-gamma)

		total_error += (cnt_potential_vehicle_collisions-len(vehicle_collisions)) ## because it returns 1 for every collision that is avoided

		return total_error/cnt_potential_vehicle_collisions


	def _compute_pedestrian_collision_error(self, cnt_potential_pedestrian_collisions, pedestrian_collisions):
		if cnt_potential_pedestrian_collisions == 0:
			return 1
		return (cnt_potential_pedestrian_collisions-len(pedestrian_collisions))/cnt_potential_pedestrian_collisions

	def _compute_static_obstacle_collision_error(self, cnt_potential_static_collisions, static_collisions, ego_vehicle_width):
		if cnt_potential_static_collisions == 0:
			return 1

		#cnt_potential_static_collisions += 1 ## to account for collisions with a building or fence

		d0 = ego_vehicle_width/2

		total_error = 0

		for col in static_collisions:
			print(col)
			obstacle_loc = np.array([col['x'], col['y']])
			ego_loc = np.array(col['ego_location'][0][:2])
			ego_velocity = np.array(col['ego_velocity'][0])
			ego_speed = calculate_speed(ego_velocity)
			dist_to_obstacle = calculate_distance(ego_loc, obstacle_loc)

			#print(ego_loc)
			#print(obstacle_loc)
			#print(ego_velocity)
			#print(dist_to_obstacle)
			
			alpha =  np.arccos(sum((ego_velocity[:2] * (obstacle_loc - ego_loc))) / dist_to_obstacle)
			#print('alpha: ', alpha)
			d1 = dist_to_obstacle * np.sin(alpha)
			#print('d1: ', d1)
			gamma = ego_speed * np.cos(alpha) * (abs(np.pi/2 - alpha)/(np.pi/2))
			#print('gamma: ', gamma)

			total_error += (d1 / d0) * math.exp(-gamma)

		#total_error += (cnt_potential_static_collisions - len(static_collisions))
		return total_error / len(static_collisions)

	def aggregate(self, ego_vehicle_length, ego_vehicle_width, results_text):
		heading_error, heading_error_of_completed_part = self._aggregate_heading_error()
		print('Heading error: ', heading_error, heading_error_of_completed_part)

		speed_error, speed_error_of_completed_part = self._aggregate_speed_error()
		print('Speed error: ', speed_error, speed_error_of_completed_part)

		path_error, path_error_of_completed_part = self._aggregate_path_error(ego_vehicle_length)
		print('Path error: ', path_error, path_error_of_completed_part)

		#potential_static_collisions, potential_pedestrian_collisions, potential_vehicle_collisions = st.session_state.carla_leaderboard_evaluator.get_total_potential_collisions()
		#print('Potential counts: ', len(potential_static_collisions), len(potential_vehicle_collisions), len(potential_pedestrian_collisions))

		print(st.session_state.carla_leaderboard_evaluator.static_collisions)

		static_col_error = self._compute_static_obstacle_collision_error(len(self.potential_static_collisions), st.session_state.carla_leaderboard_evaluator.static_collisions, ego_vehicle_width)
		print('Static col error: ', static_col_error)

		vehicle_col_error = self._compute_vehicle_collision_error(len(self.potential_vehicle_collisions)-1, st.session_state.carla_leaderboard_evaluator.vehicle_collisions, ego_vehicle_width)
		print('Vehicle col error: ', vehicle_col_error)

		pedestrian_col_error = self._compute_pedestrian_collision_error(len(self.potential_pedestrian_collisions), st.session_state.carla_leaderboard_evaluator.pedestrian_collisions)
		print('Pedestrian col error: ', pedestrian_col_error)

		results_text.append(['Speed Deviation', str(speed_error_of_completed_part), str(round(speed_error, 3))])
		results_text.append(['Heading Deviation', str(heading_error_of_completed_part), str(round(heading_error, 3))])
		results_text.append(['Path Deviation', str(path_error_of_completed_part), str(round(path_error, 3))])
		results_text.append(['Static Obstacle Collision', '{}/{}'.format(len(st.session_state.carla_leaderboard_evaluator.static_collisions), len(self.potential_static_collisions)+1), str(round(static_col_error, 3))])
		results_text.append(['Dynamic Obstacle Collision', '{}/{}'.format(len(st.session_state.carla_leaderboard_evaluator.vehicle_collisions), len(self.potential_vehicle_collisions)+1), str(round(vehicle_col_error, 3))])
		results_text.append(['Pedestrian Collision', '{}/{}'.format(len(st.session_state.carla_leaderboard_evaluator.pedestrian_collisions), len(self.potential_pedestrian_collisions)), str(round(pedestrian_col_error, 3))])

		with open('{}/aggregate_report.pkl'.format(st.session_state.results_folder_path), 'wb') as f:
			pickle.dump(results_text, f)

		output = "\n"
		output += tabulate(results_text, tablefmt='fancy_grid')
		output += "\n"
		print(output)
