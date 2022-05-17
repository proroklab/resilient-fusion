from __future__ import print_function
import math
import signal
import sys
import time
import numpy as np
import py_trees
import carla
import matplotlib.pyplot as plt

from carlaviz.examples.carla_painter import CarlaPainter
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

import xml.etree.ElementTree as ET

import pickle

def is_ego_in_section(section_coord1, section_coord2, ego_coord):
    def check_coord_range(c1, c2, ego_c):
        c1_tokens = str(c1).split('.')
        c1 = float('.'.join((c1_tokens[0], c1_tokens[1][:3])))

        c2_tokens = str(c2).split('.')
        c2 = float('.'.join((c2_tokens[0], c2_tokens[1][:3])))

        ego_tokens = str(ego_c).split('.')
        ego_c = float('.'.join((ego_tokens[0], ego_tokens[1][:3])))

        min_coord = min(c1, c2, c1-4, c1+4, c2-4, c2+4)
        max_coord = max(c1, c2, c1-4, c1+4, c2-4, c2+4)
        ego_coord = ego_c
        print(min_coord, ego_coord, max_coord, min_coord <= ego_coord <= max_coord)
        return min_coord <= ego_coord <= max_coord
        
    print('check coord x: ', end=' ')
    x_in_range = check_coord_range(section_coord1[0], section_coord2[0], ego_coord[0])
    print('check coord y: ', end=' ')
    y_in_range = check_coord_range(section_coord1[1], section_coord2[1], ego_coord[1])
    return x_in_range and y_in_range


class Visulization(object):
    def __init__(self, routes_filepath, world):
        self.painter = CarlaPainter('localhost', 8089)
        self.painter2 = CarlaPainter('localhost', 8089)
        self.painter3 = CarlaPainter('localhost', 8089)
        self.test_timeout = 0

        self.actual_traj = [[]]

        self.traj_major_points = self._parse_expected_trajectory(routes_filepath)
        print('Traj major points: ', self.traj_major_points)

        self.expected_traj = self._get_dense_trajectory(world)
        self.traj_major_points_idx = 0
        self.current_section = 1
        self.first_step = True

    def _get_dense_trajectory(self, world):
        route_planner_dao = GlobalRoutePlannerDAO(wmap=world.get_map(), sampling_resolution=2)
        route_planner_agent = GlobalRoutePlanner(dao=route_planner_dao)
        route_planner_agent.setup()

        expected_wps = []

        for i in range(len(self.traj_major_points[0])-1): 
            pt1 = self.traj_major_points[0][i]
            pt2 = self.traj_major_points[0][i+1]

            loc1 = carla.Location(x=pt1[0], y=pt1[1], z=pt1[2])
            loc2 = carla.Location(x=pt2[0], y=pt2[1], z=pt2[2])

            wps = route_planner_agent.trace_route(origin=loc1, destination=loc2)
            expected_wps += wps
        
        expected_traj = [[wp.transform.location.x, wp.transform.location.y, wp.transform.location.z] for wp, _ in expected_wps]

        with open('/home/saashanair/toolkit_STABLE/dense_traj.pkl', 'wb') as f:
            pickle.dump(expected_traj, f)

        return [expected_traj]

    ## TODO: accept route index in the UI to then send as an argument to this function
    def _parse_expected_trajectory(self, routes_filepath, route_index=0):
        traj_major_points = []

        root = ET.parse(routes_filepath).getroot()
        for i in range(len(root[route_index])): 
            loc_attrib = root[route_index][i].attrib
            print(i, loc_attrib)
            traj_major_points.append([float(loc_attrib['x']), float(loc_attrib['y']), float(loc_attrib['z'])])

        return [traj_major_points]




    '''
    ## TODO: accept route index in the UI to then send as an argument to this function
    def _parse_expected_trajectory(self, routes_filepath, world, route_index=0):
        expected_traj = []

        route_planner_dao = GlobalRoutePlannerDAO(wmap=world.get_map(), sampling_resolution=2)
        route_planner_agent = GlobalRoutePlanner(dao=route_planner_dao)
        route_planner_agent.setup()

        expected_wps = []

        root = ET.parse(routes_filepath).getroot()
        for i in range(len(root[route_index])-1): 
            loc1_attrib = root[route_index][i].attrib
            loc2_attrib = root[route_index][i+1].attrib

            loc1 = carla.Location(x=float(loc1_attrib['x']), y=float(loc1_attrib['y']), z=float(loc1_attrib['z']))
            loc2 = carla.Location(x=float(loc2_attrib['x']), y=float(loc2_attrib['y']), z=float(loc2_attrib['z']))

            wps = route_planner_agent.trace_route(origin=loc1, destination=loc2)
            expected_wps += wps
        
        expected_traj = [[wp.transform.location.x, wp.transform.location.y, wp.transform.location.z] for wp, _ in expected_wps]

        return [expected_traj]
    '''

    

    def _tick_(self,ego_vehicles):

            ego_location = ego_vehicles[0].get_location()
            ego_velocity = ego_vehicles[0].get_velocity()

            if self.first_step: ## if this is the first time step, ensure that the traj starts from the spawn location of the ego
                self.traj_major_points[0][0] = [ego_location.x, ego_location.y, ego_location.z]
                self.first_step = False

            current_section_start_coords = self.traj_major_points[0][self.traj_major_points_idx]
            current_section_start_loc = carla.Location(x=current_section_start_coords[0], y=current_section_start_coords[1], z=ego_location.z)

            current_section_end_coords = self.traj_major_points[0][self.traj_major_points_idx+1]
            current_section_end_loc = carla.Location(x=current_section_end_coords[0], y=current_section_end_coords[1], z=ego_location.z)

            ego_coord = [ego_location.x, ego_location.y, ego_location.z]
            if not is_ego_in_section(current_section_start_coords, current_section_end_coords, ego_coord):
                self.current_section += 1
                self.traj_major_points_idx += 1

            loc_str = 'Idx: {}; Section: {} \n Start: {} \n Actual: {} \n End: {}'.format(self.traj_major_points_idx, self.current_section, current_section_start_loc, ego_location, current_section_end_loc)
            print(loc_str)


            self.actual_traj[0].append([ego_location.x, ego_location.y, ego_location.z])

            print(len(self.actual_traj), len(self.expected_traj), len(self.traj_major_points))
            print(len(self.actual_traj[0]), len(self.expected_traj[0]), len(self.traj_major_points[0]))
            print(len(self.actual_traj[0][0]), len(self.expected_traj[0][0]), len(self.traj_major_points[0][0]))

            # draw trajectories
            self.painter.draw_polylines(self.actual_traj, color='#FF0000')
            self.painter2.draw_polylines(self.expected_traj, color='#0000FF')
            self.painter3.draw_points(self.traj_major_points)
            self.painter3.draw_texts([loc_str], [[ego_location.x, ego_location.y + 20, ego_location.z + 5.0]], size=20)