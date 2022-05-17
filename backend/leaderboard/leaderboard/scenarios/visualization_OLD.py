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
class Visulization(object):
    def __init__(self):
        self.painter = CarlaPainter('localhost', 8089)
        self.painter2 = CarlaPainter('localhost', 8089)
        self.test_timeout = 0

        self.green_line = [[],[]] # Planned trajectory
        self.green_line[0].append([-78, -10, 0.03])
        self.green_line[0].append([-78, -77, 0.03])
        self.green_line[0].append([-74.5, -87, 0.03])
        self.green_line[0].append([-74.5, -128, 0.03])
        for i in range(100):
            self.green_line[0].append([-66.5 - 8 * np.cos(np.pi*i / 200), -128 - 8 * np.sin(np.pi*i / 200), 0.03])
        self.green_line[0].append([-66.5, -136, 0.03])
        self.green_line[0].append([-30, -135.5, 0.03])

        plt.ion()
        # time.sleep(15)
        plt.rcParams['figure.figsize'] = (5, 2)

        self.init_variable()
    def init_variable(self):
        self.trajectories = [[]]

        self.state = 1
        self.step = 1

        self.L = 0

        self.Y_rad = 0
        self.Y_speed = 0
        self.last_y = 0
        self.last_x = 0

        self.x_L = []
        self.y_rad = []
        self.y_speed = []
        self.y_speed_green = []
        self.y_rad_green = []

    def _tick_(self,ego_vehicles):

            ego_location = ego_vehicles[0].get_location()
            ego_velocity = ego_vehicles[0].get_velocity()


            self.trajectories[0].append([ego_location.x, ego_location.y, ego_location.z])

            # draw trajectories
            self.painter.draw_polylines(self.trajectories, color='#FF0000')
            self.painter2.draw_polylines(self.green_line, color='#00FF00')

            self.Y_rad = np.arccos(ego_velocity.x / (0.001 + np.sqrt(ego_velocity.x ** 2 + ego_velocity.y ** 2)))
            self.Y_speed = np.sqrt(ego_velocity.x ** 2 + ego_velocity.y ** 2)

            ero_rad = 0
            ero_speed = 0
            ero_path = 0

            if -77 <= ego_location.y <= -10 and -80 <= ego_location.x <= -72:
                last_L = self.L
                self.L = -10 - ego_location.y
                ero_rad = self.Y_rad - np.pi / 2
                ero_speed = self.Y_speed - 6
                ero_path = ego_location.x + 78
                self.x_L.append(self.L)
                self.y_rad.append(self.Y_rad)
                self.y_speed.append(self.Y_speed)
                self.y_rad_green.append(np.pi / 2)
                self.y_speed_green.append(6)


            if -87 <= ego_location.y < -77 and -80 <= ego_location.x <= -72:
                last_L = self.L
                self.L = 67 + (-77 - ego_location.y)*np.sqrt(100+3.5**2)/10
                ero_rad = self.Y_rad - np.arccos(3.5/np.sqrt(100+3.5**2)/10)
                ero_speed = self.Y_speed - 6
                ero_path = ego_location.x + (78-0.35*(-77-ego_location.y))
                self.x_L.append(self.L)
                self.y_rad.append(self.Y_rad)
                self.y_speed.append(self.Y_speed)
                self.y_rad_green.append(np.arccos(3.5/np.sqrt(100+3.5**2)/10))
                self.y_speed_green.append(6)


            L0 = 67 + np.sqrt(100+3.5**2)


            if -128 <= ego_location.y < -87 and -80 <= ego_location.x <= -72:
                last_L = self.L
                self.L = L0 -87 - ego_location.y
                ero_rad = self.Y_rad - np.pi / 2
                ero_speed = self.Y_speed - 6
                ero_path = ego_location.x + 74.5
                self.x_L.append(self.L)
                self.y_rad.append(self.Y_rad)
                self.y_speed.append(self.Y_speed)
                self.y_rad_green.append(np.pi / 2)
                self.y_speed_green.append(6)


            L1 = L0 + 41

            if -140 <= ego_location.y < -128 and -78 <= ego_location.x < -66.5:
                thea_turn = np.pi / 2 - np.arccos(-(ego_location.x+66.5)/np.sqrt((ego_location.x+66.5)**2+(ego_location.y+128)**2))
                last_L = self.L
                self.L = L1 + 8*np.arccos(-(ego_location.x+66.5)/np.sqrt((ego_location.x+66.5)**2+(ego_location.y+128)**2))
                ero_rad = self.Y_rad - thea_turn
                ero_speed = self.Y_speed - 6
                ero_path = (ego_location.y + 128 + 8*np.sin(np.pi/2-thea_turn))/np.cos(thea_turn)
                self.x_L.append(self.L)
                self.y_rad.append(self.Y_rad)
                self.y_speed.append(self.Y_speed)
                self.y_rad_green.append(thea_turn)
                self.y_speed_green.append(6)

            L2 = L1 + 4*np.pi

            if -66.5 <= ego_location.x <= -30:
                last_L = self.L
                self.L = L2 + ego_location.x + 66.5
                ero_rad = self.Y_rad
                ero_speed = self.Y_speed - 6
                ero_path = ego_location.y + 135.5
                self.x_L.append(self.L)
                self.y_rad.append(self.Y_rad)
                self.y_speed.append(self.Y_speed)
                self.y_rad_green.append(0)
                self.y_speed_green.append(6)


            velocity_str = "error_rad: {:.2f}".format(ero_rad) + ", error_speed: {:.2f}".format(ero_speed) + ", error_path: {:.2f}".format(ero_path)
            self.painter.draw_texts([velocity_str], [[ego_location.x, ego_location.y + 20, ego_location.z + 5.0]], size=20)

            self.last_y = ego_location.y
            self.last_x = ego_location.x


            plt.clf()
            fig1_rad = plt.subplot(1, 2, 1)
            fig1_rad.plot(self.x_L[20:-1], self.y_rad[20:-1], color='red', linewidth=1.0, linestyle='-')
            fig1_rad.plot(self.x_L[20:-1], self.y_rad_green[20:-1], color='green', linewidth=1.0, linestyle='-')

            fig2_speed = plt.subplot(1, 2, 2)
            fig2_speed.plot(self.x_L[20:-1], self.y_speed[20:-1], color='red', linewidth=1.0, linestyle='-')
            fig2_speed.plot(self.x_L[20:-1], self.y_speed_green[20:-1], color='green', linewidth=1.0,
                                    linestyle='-')
            plt.pause(0.1)


