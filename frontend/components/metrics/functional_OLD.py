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

ROUTE_TO_SPEED_MAPPING = {
	'additional_routes/routes_town03_long.xml': [6, 3, 6, 2, 5], 
	'validation_routes/routes_town05_short.xml': [3],
	'validation_routes/routes_town05_tiny.xml': []
}

ROUTE_TO_K_MAPPING = {
	'additional_routes/routes_town03_long.xml': 3, 
	'validation_routes/routes_town05_short.xml': 1,
	'validation_routes/routes_town05_tiny.xml': 3
}

ROUTE_TO_N_MAPPING = {}

ROUTE_TO_M_MAPPING = {}

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

'''
def read_expert_data():
	results_folder = st.session_state.results_folder_path.split('/')
	results_folder[-1] = 'expert' ## read the expert folder given the current route and scenario settings
	results_folder.append('ego_stats.csv')
	results_folder = '/'.join(results_folder)
	#print('results_folder: ', results_folder)
	expert_data_file = Path(results_folder)
	if not expert_data_file.exists():
		st.error('Expert data does not exist for this route/scenario. Please collect needed data.')
		return

	return pd.read_csv(expert_data_file)
'''

class DrivingMetrics:
	def __init__(self, baseline_traj, baseline_speed, route_path):
		self.baseline_traj_with_lanetypes = baseline_traj

		self.baseline_traj = self._extract_baseline_traj_wps()
		#self.baseline_speed = baseline_speed

		self.sparse_traj = self._read_route_file('/'.join((os.environ['LEADERBOARD_ROUTE_ROOT'], route_path)))
		self.baseline_traj_section_mapping = self._determine_end_points_of_route_section()

		#print('BASELINE TRAJ: ', self.baseline_traj_section_mapping)
		self.total_sections = len(self.sparse_traj)-1 ## coz two points together make one section

		self.K = ROUTE_TO_K_MAPPING[route_path]
		#self.baseline_speed_per_section = [6, 3, 6, 2, 5]
		self.baseline_speed_per_section = ROUTE_TO_SPEED_MAPPING[route_path]
		if len(self.baseline_speed_per_section) == 0:
			self.baseline_speed_per_section = [2] * self.total_sections

		self.heading_actual = []
		self.heading_expected = []
		self.speed_actual = []
		self.speed_expected = []
		
		self.heading_error_per_section = [0] * self.total_sections
		self.speed_error_per_section = [0] * self.total_sections
		self.path_error_per_section = [0] * self.total_sections

		self.L = 0
		self.last_L = [0] * self.total_sections
		self.last_projected_loc = self.baseline_traj[0]

		self.step_cntr = 0

		self.prev_loc = None

	def _extract_baseline_traj_wps(self):
		dense_traj = []
		for i, r in enumerate(self.baseline_traj_with_lanetypes):
			#print('DT: ', i, r)
			l = r[0].location
			dense_traj.append([l.x, l.y])
		return np.array(dense_traj)

	def _read_route_file(self,route_path, route_index=0):
		traj_major_points = []

		root = ET.parse(route_path).getroot()
		for i in range(len(root[route_index])): 
			loc_attrib = root[route_index][i].attrib
			traj_major_points.append([float(loc_attrib['x']), float(loc_attrib['y'])])

		return np.array(traj_major_points)

	def _project_to_route(self, loc_xy):
		dist_to_route_wps = np.sum((self.baseline_traj - loc_xy)**2, axis=1)
		#print('Dist to route wps: ', dist_to_route_wps)
		idx_closest_wp = np.argmin(dist_to_route_wps)
		return self.baseline_traj[idx_closest_wp].tolist(), idx_closest_wp, dist_to_route_wps[idx_closest_wp]

	def _determine_end_points_of_route_section(self):
		temp = []
		for pt in self.sparse_traj:
			_, idx, _ = self._project_to_route(pt)
			temp.append(idx)

		return temp

	def _get_current_route_section(self, projected_loc_idx):
		#print('In get section: ', self.baseline_traj_section_mapping, projected_loc_idx)
		## find where in the baseline_traj_section_mapping the idx of the projected location lies, this allows to determine the section of the route that the location falls within
		projected_loc_in_section_idx = bisect.bisect(self.baseline_traj_section_mapping, projected_loc_idx)
		section_number = min(max(0, projected_loc_in_section_idx-1), self.total_sections-1) ## the element before and after the position of the projected_loc_idx are the lower and higher ends of the respective section 
		return section_number, self.baseline_traj_section_mapping[section_number]


	#def _compute_expected_heading(self, loc1, loc2):
	#	return math.atan2(loc2[1]-loc1[1], loc2[0]-loc1[0])

	def _compute_expected_heading(self, projected_loc_idx):
		h = 0
		projected_loc_idx_curr = projected_loc_idx

		cnt = 1

		for i in range(cnt):
			projected_loc_idx_next = min(projected_loc_idx_curr+2, len(self.baseline_traj)-1)
			#projected_loc_idx_next = max(projected_loc_idx_curr-1, 0)
			loc1 = self.baseline_traj[projected_loc_idx_curr]
			loc2 = self.baseline_traj[projected_loc_idx_next]
			vec = loc2-loc1
			h += np.arccos(vec[0] / (0.001 + np.sqrt(vec[0] ** 2 + vec[1] ** 2)))
			#h += math.atan2(loc2[0]-loc1[0], loc2[1]-loc1[1])
			#h += math.atan2(loc2[1]-loc1[1], loc2[0]-loc1[0])

			#X = math.cos(loc2[0]) * math.sin(loc2[1]-loc1[1])
			#Y = math.cos(loc2[0]) * math.sin(loc1[0]) - math.sin(loc2[0]) * math.sin(loc1[0]) * math.cos(loc2[1]-loc1[1])
			#h += math.atan2(Y, X)
			print(i, h)
			projected_loc_idx_curr = projected_loc_idx_next

		print('h: ', h/cnt)
		return h/cnt

		
	def _compute_L(self, start_idx, end_idx):
		#print('In compute L: ', start_idx, end_idx)
		left_pts = self.baseline_traj[start_idx: end_idx]
		right_pts = self.baseline_traj[start_idx+1: end_idx+1]
		#print('Pts lists: ', left_pts.shape, right_pts.shape)

		#diff = left_pts - right_pts
		#print('Diff: ', diff)
		dist = np.linalg.norm(left_pts - right_pts)
		#print('Dist: ', dist)
		return dist
		

	def _compute_heading_error(self, ego_heading, projected_loc_idx, current_L, last_L_in_section):
		self.heading_actual.append(ego_heading)
		if projected_loc_idx >= len(self.baseline_traj)-2:
			projected_loc_idx = len(self.baseline_traj) - 3
		#expected_heading = self._compute_expected_heading(self.baseline_traj[projected_loc_idx], self.baseline_traj[projected_loc_idx+1])
		#expected_heading = self._compute_expected_heading(self.baseline_traj[projected_loc_idx], self.baseline_traj[projected_loc_idx+2])
		expected_heading = self._compute_expected_heading(projected_loc_idx)
		st.session_state.carla_ego_stats.loc[st.session_state.carla_ego_stats.index[-1], 'expected_heading'] = expected_heading
		#st.session_state.carla_ego_stats.loc[st.session_state.carla_ego_stats.index[-1], 'expected_heading2'] = expected_heading2
		print('Heading: ', expected_heading, ego_heading) #, st.session_state.carla_expert_stats['heading'][projected_loc_idx])

		heading_error = ego_heading - expected_heading
		###self.E_heading = self.E_heading + (self.L - last_L) * abs(self.Y_rad - np.pi / 2)
		return (current_L - last_L_in_section) * abs(heading_error)

	'''
	def _compute_speed_error(self, ego_speed, current_baseline_speed, current_L, last_L_in_section):
		speed_error = ego_speed - current_baseline_speed
		st.session_state.carla_ego_stats.loc[st.session_state.carla_ego_stats.index[-1], 'expected_speed'] = current_baseline_speed
		#print('Speed: ', ego_speed, current_baseline_speed, speed_error)
		return (current_L - last_L_in_section) * abs(speed_error)
	'''

	def _compute_expected_speed(self, projected_loc_idx, is_at_traffic_light):
		'''
		print(projected_loc_idx, self.baseline_traj_with_lanetypes[projected_loc_idx][1], self.baseline_traj_with_lanetypes[min(projected_loc_idx, projected_loc_idx+2)][1])
		if self.baseline_traj_with_lanetypes[min(projected_loc_idx, projected_loc_idx+2)][1] in [RoadOption.STRAIGHT, RoadOption.LANEFOLLOW]:
			return 6
		else:
			return 2
		'''

		projected_loc_idx_prev = max(0, projected_loc_idx-1)
		projected_loc_idx_prev_2 = max(0, projected_loc_idx-2)
		projected_loc_idx_next = min(projected_loc_idx, projected_loc_idx+1)
		projected_loc_idx_next_2 = min(projected_loc_idx, projected_loc_idx+2)
		roadoptions_of_wps_of_interest = set([self.baseline_traj_with_lanetypes[projected_loc_idx_prev_2][1],
												self.baseline_traj_with_lanetypes[projected_loc_idx_prev][1], 
												self.baseline_traj_with_lanetypes[projected_loc_idx][1], 
												self.baseline_traj_with_lanetypes[projected_loc_idx_next][1],
												self.baseline_traj_with_lanetypes[projected_loc_idx_next_2][1]])
		#print('YO: ', roadoptions_of_wps_of_interest, is_at_traffic_light)

		if is_at_traffic_light:
			return 0

		if roadoptions_of_wps_of_interest.intersection([RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT, RoadOption.RIGHT, RoadOption.LEFT]):
			return 2
		return 6
		

	def _compute_speed_error(self, ego_speed, projected_loc_idx, current_L, last_L_in_section, is_at_traffic_light):
		self.speed_actual.append(ego_speed)
		expected_speed = self._compute_expected_speed(projected_loc_idx, is_at_traffic_light)
		st.session_state.carla_ego_stats.loc[st.session_state.carla_ego_stats.index[-1], 'expected_speed'] = expected_speed

		speed_error = ego_speed - expected_speed
		return (current_L - last_L_in_section) * abs(speed_error)


	def _compute_path_error(self, dist_to_projected_loc, current_projected_loc, previous_projected_loc, current_L, last_L_in_section):
		dist_to_previous_projected_loc = np.linalg.norm(current_projected_loc - previous_projected_loc)
		#print('Dists prev and curr: ', dist_to_previous_projected_loc, dist_to_projected_loc)
		#print('L diff: ', current_L - last_L_in_section)
		return dist_to_previous_projected_loc * dist_to_projected_loc


	def compute(self, ego_location, ego_heading, ego_speed, is_at_traffic_light):
		if self.prev_loc is None:
			self.prev_loc = ego_location
		ego_heading_by_loc = math.atan2(ego_location[1]-self.prev_loc[1], ego_location[0]-self.prev_loc[0])
		st.session_state.carla_ego_stats.loc[st.session_state.carla_ego_stats.index[-1], 'ego_heading_by_loc'] = ego_heading_by_loc



		projected_loc, projected_loc_idx, dist_to_projected_loc = self._project_to_route(np.array([ego_location[0], ego_location[1]]))
		st.session_state.carla_ego_stats.loc[st.session_state.carla_ego_stats.index[-1], 'projected_x'] = projected_loc[0]
		st.session_state.carla_ego_stats.loc[st.session_state.carla_ego_stats.index[-1], 'projected_y'] = projected_loc[1]
		
		#print('Projected: ', projected_loc, projected_loc_idx, dist_to_projected_loc)
		section_number, section_start_idx = self._get_current_route_section(projected_loc_idx)
		#print('Section: ', section_number, section_start_idx)
		current_L = self._compute_L(section_start_idx, projected_loc_idx)
		st.session_state.carla_ego_stats.loc[st.session_state.carla_ego_stats.index[-1], 'L'] = current_L
		#print('Current L: ', current_L, self.last_L)

		current_heading_error = self._compute_heading_error(ego_heading, projected_loc_idx, current_L, self.last_L[section_number])
		#print('Current heading error: ', current_heading_error)

		self.heading_error_per_section[section_number] += current_heading_error
		#print('Heading error per section: ', self.heading_error_per_section)

		#current_speed_error = self._compute_speed_error(ego_speed, self.baseline_speed_per_section[section_number], current_L, self.last_L[section_number])
		current_speed_error = self._compute_speed_error(ego_speed, projected_loc_idx, current_L, self.last_L[section_number], is_at_traffic_light)
		#print('Current speed error: ', current_speed_error)
		self.speed_error_per_section[section_number] += current_speed_error
		#print('Speed error per section: ', self.speed_error_per_section)

		current_path_error = self._compute_path_error(dist_to_projected_loc, projected_loc, self.last_projected_loc, current_L, self.last_L[section_number])
		#print('Current path error: ', current_path_error)
		self.path_error_per_section[section_number] += current_path_error
		#print('Path error per section: ', self.path_error_per_section)


		#print('Speed: ', ego_speed, st.session_state.carla_expert_stats['speed'][projected_loc_idx])

		self.last_projected_loc = np.array(projected_loc)
		#print('Update last project loc: ', self.last_projected_loc)
		self.last_L[section_number] = current_L
		#print('Updated last L: ', self.last_L)
		#print('DF row: ', st.session_state.carla_ego_stats.iloc[-1])

	def _compute_avg_expected_heading_for_route_section(self, section_number):
		section_start_idx = self.baseline_traj_section_mapping[section_number]
		section_end_idx = self.baseline_traj_section_mapping[section_number+1]

		total_expected_heading = 0
		N = 0

		for idx in range(section_start_idx, section_end_idx):
			total_expected_heading += self._compute_expected_heading(self.baseline_traj[idx], self.baseline_traj[idx+1])
			N += 1

		#print('Total expected heading: ', section_number, total_expected_heading, N)
		return total_expected_heading/N

	def _compute_avg_expected_speed_for_route_section(self, section_number):
		speed_for_section = self.baseline_speed_per_section[section_number]

		#section_start_idx = self.baseline_traj_section_mapping[section_number]
		#section_end_idx = self.baseline_traj_section_mapping[section_number+1]
		#N = section_end_idx - section_start_idx
		#avg = (speed_for_section * N) / N
		## since it is constant speed, the avg speed would be (speed * N) / N
		
		return speed_for_section

	def _compute_L_per_route_section(self):
		L_per_section = []

		for section_number in range(self.total_sections):
			section_start_idx = self.baseline_traj_section_mapping[section_number]
			section_end_idx = self.baseline_traj_section_mapping[section_number+1]

			L_per_section.append(self._compute_L(section_start_idx, section_end_idx))

		return L_per_section

	def _aggregate_heading_error(self, total_L_per_section):
		for idx in range(self.total_sections):
			avg_heading = self._compute_avg_expected_heading_for_route_section(idx)
			#print('Avg heading: ', avg_heading)
			normalized_error = self.heading_error_per_section[idx] / (total_L_per_section[idx] * avg_heading)
			#print('Norm error: ', normalized_error)
			self.heading_error_per_section[idx] = math.exp(-normalized_error)
			#print('Updated heading error: ', self.heading_error_per_section)

		return np.sum(self.heading_error_per_section)/(self.K)

	def _aggregate_speed_error(self, total_L_per_section):
		for idx in range(self.total_sections):
			avg_speed = self._compute_avg_expected_speed_for_route_section(idx)
			#print('Avg speed: ', avg_speed)
			normalized_error = self.speed_error_per_section[idx] / (total_L_per_section[idx] * avg_speed)
			#print('Norm error: ', normalized_error)
			self.speed_error_per_section[idx] = math.exp(-normalized_error)

		return np.sum(self.speed_error_per_section)/(self.K)

	def _aggregate_path_error(self, total_L_per_section, ego_vehicle_length):
		for idx in range(self.total_sections):
			normalized_error = self.path_error_per_section[idx] / (total_L_per_section[idx] * ego_vehicle_length)
			#print('Norm error: ', normalized_error)
			self.path_error_per_section[idx] = math.exp(-normalized_error)

		return np.sum(self.path_error_per_section)/(self.K)

	'''
	def _get_all_collisions(self):
		static_cols = []
		pedestrian_cols = []
		vehicle_cols = []

		for node in st.session_state.carla_statistics_manager._master_scenario.get_criteria():
			if node.list_traffic_events:
				for event in node.list_traffic_events:
					if event.get_type() == TrafficEventType.COLLISION_STATIC:
						static_cols.append(event.get_dict())
					elif event.get_type() == TrafficEventType.COLLISION_VEHICLE:
						vehicle_cols.append(event.get_dict())
					elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN:
						pedestrian_cols.append(event.get_dict())

		return static_cols, pedestrian_cols, vehicle_cols
	'''

	def _compute_static_obstacle_collision_error(self, static_collisions, ego_vehicle_width):
		if len(static_collisions) == 0:
			return 0

		d0 = ego_vehicle_width/2

		total_error = 0

		for col in static_collisions:
			obstacle_loc = np.array([col['x'], col['y']])
			ego_loc = np.array(col['ego_location'][0][:2])
			ego_velocity = np.array(col['ego_velocity'][0])
			ego_speed = calculate_speed(ego_velocity)
			dist_to_obstacle = calculate_distance(ego_loc, obstacle_loc)
			
			alpha =  np.arccos((ego_velocity * (obstacle_loc - ego_loc)) / dist_to_obstacle)
			d1 = dist_to_obstacle * np.sin(alpha)
			gamma = ego_speed * np.cos(alpha) * (abs(np.pi/2 - alpha)/(np.pi/2))

			total_error += (d1 / d0) * math.exp(-gamma)

		## TODO: normalize the sum
		return total_error / len(static_collisions)


	def _compute_pedestrian_collision_error(self, pedestrian_collisions, potential_pedestrian_collisions):
		if potential_pedestrian_collisions == 0:
			return 0
		total_error = len(pedestrian_collisions)/potential_pedestrian_collisions

		## TODO: normalize the sum
		return total_error


	def _compute_vehicle_collision_error(self, vehicle_collisions, ego_vehicle_width):
		if len(vehicle_collisions) == 0:
			return 0

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

			print('d0: ', d0)
			print('d1: ', d1)
			print('alpha: ', alpha)
			print('gamma: ', gamma)

			total_error += (d1 / d0) * math.exp(-gamma)

		return total_error/len(vehicle_collisions)


	def aggregate(self, ego_vehicle_length, ego_vehicle_width, results_text):
		total_L_per_section = self._compute_L_per_route_section()
		aggregate_heading_error = self._aggregate_heading_error(total_L_per_section)
		print('Aggregate heading error: ', aggregate_heading_error)
		aggregate_speed_error = self._aggregate_speed_error(total_L_per_section)
		print('Aggregate speed error: ', aggregate_speed_error)
		#print('Ego vehicle length: ', ego_vehicle_length)
		aggregate_path_error = self._aggregate_path_error(total_L_per_section, ego_vehicle_length)
		print('Aggregate path error: ', aggregate_path_error)

		print('Static collisions: ', st.session_state.carla_leaderboard_evaluator.static_collisions)
		print('Pedestrian collisions: ', st.session_state.carla_leaderboard_evaluator.pedestrian_collisions)
		print('Vehicle collisions: ', st.session_state.carla_leaderboard_evaluator.vehicle_collisions)

		potential_static_collisions, potential_pedestrian_collisions, potential_vehicle_collisions = st.session_state.carla_leaderboard_evaluator.get_total_potential_collisions()
		print('Total potential collisions: ', potential_static_collisions, potential_pedestrian_collisions, potential_vehicle_collisions)
		
		static_collision_error = self._compute_static_obstacle_collision_error(st.session_state.carla_leaderboard_evaluator.static_collisions, ego_vehicle_width)
		print('Static collision error: ', static_collision_error)

		pedestrian_collision_error = self._compute_pedestrian_collision_error(st.session_state.carla_leaderboard_evaluator.pedestrian_collisions, potential_pedestrian_collisions)
		print('Pedestrian collision error: ', pedestrian_collision_error)

		vehicle_collision_error = self._compute_vehicle_collision_error(st.session_state.carla_leaderboard_evaluator.vehicle_collisions, ego_vehicle_width)
		print('Vehicle collision error: ', vehicle_collision_error)

		results_text.append(['Speed Deviation', '', str(round(aggregate_speed_error, 3))])
		results_text.append(['Heading Deviation', '', str(round(aggregate_heading_error, 3))])
		results_text.append(['Path Deviation', '', str(round(aggregate_path_error, 3))])
		results_text.append(['Static Obstacle Collision', '', str(round(static_collision_error, 3))])
		results_text.append(['Dynamic Obstacle Collision', '', str(round(vehicle_collision_error, 3))])
		results_text.append(['Pedestrian Collision', '', str(round(pedestrian_collision_error, 3))])

		with open('{}/aggregate_report.pkl'.format(st.session_state.results_folder_path), 'wb') as f:
			pickle.dump(results_text, f)

		output = "\n"
		output += tabulate(results_text, tablefmt='fancy_grid')
		output += "\n"
		print(output)




''' Heading, Speed, Path errors

## Additional route / Town03
1. expert: 16.31223099073402, 1.0068686723068976, 1.1928762064439558
2. transfuser: 16.081369460405885, 1.0164738925941956, 0.9672627492560006
3. CIL: 4.979872603410667, 1.3699438343033936, 0.9709782966257706 [Projected dist for off path = 199]
However the values for CIL make no sense as the route was not completed, the agent drove off the path and just stayed there!
Could we use the distance to the projected point to determine failure case?
What do we do if the agent drives off path but then eventually comes back to the path?

## Town05 short
1. expert: 0.923333681380098, 0.4887203573665887, 0.7479311561453745
2. transfuser: 0.9143526789666178, 0.5447557574780135, 0.5009271069884716
3. CIL: 0.9099270280160562, 0.8835329514264403, 0.511599246965106 [Projected distance for off path = 12]

'''
















