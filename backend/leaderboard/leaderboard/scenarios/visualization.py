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


class Visulization(object):
    def __init__(self, expected_traj):
        self.painter = CarlaPainter('localhost', 8089)
        self.painter2 = CarlaPainter('localhost', 8089)
        self.test_timeout = 0

        self.actual_traj = [[]]
        self.expected_traj = [[pt[0].location.x, pt[0].location.y, 0.0] for pt in expected_traj] ## expected traj is of the form (wp, lanetype)
    

    def _tick_(self, ego_vehicles):

            ego_location = ego_vehicles[0].get_location()

            self.actual_traj[0].append([ego_location.x, ego_location.y, ego_location.z])

            # draw trajectories
            self.painter2.draw_polylines([self.expected_traj], color='#00FF00', width=0.5)
            self.painter.draw_polylines(self.actual_traj, color='#FF0000', width=0.5)