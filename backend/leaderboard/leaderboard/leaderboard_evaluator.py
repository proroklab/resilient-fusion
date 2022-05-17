#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator Routes

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""
from __future__ import print_function

import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
from distutils.version import LooseVersion
import importlib
import os
import sys
import gc
import pkg_resources
import sys
import carla
import copy
#import signal
import time
import numpy as np

import srunner
from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog
from srunner.scenariomanager.traffic_events import TrafficEventType

from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.envs.sensor_interface import SensorInterface, SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper import  AgentWrapper, AgentError
from leaderboard.utils.statistics_manager import StatisticsManager
from leaderboard.utils.route_indexer import RouteIndexer


sensors_to_icons = {
    'sensor.camera.rgb':        'carla_camera',
    'sensor.camera.semantic_segmentation': 'carla_camera',
    'sensor.camera.depth':      'carla_camera',
    'sensor.lidar.ray_cast':    'carla_lidar',
    'sensor.lidar.ray_cast_semantic':    'carla_lidar',
    'sensor.other.radar':       'carla_radar',
    'sensor.other.gnss':        'carla_gnss',
    'sensor.other.imu':         'carla_imu',
    'sensor.opendrive_map':     'carla_opendrive_map',
    'sensor.speedometer':       'carla_speedometer',
    'sensor.other.obstacle':    'carla_radar',
}


class LeaderboardEvaluator(object):

    """
    TODO: document me!
    """

    ego_vehicles = []

    # Tunable parameters
    client_timeout = 10.0  # in seconds
    wait_for_world = 20.0  # in seconds
    frame_rate = 20.0      # in Hz

    def __init__(self, args, statistics_manager):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """
        self.statistics_manager = statistics_manager
        self.sensors = None
        self.sensor_icons = []
        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam

        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        self.client = carla.Client(args.host, int(args.port))
        if args.timeout:
            self.client_timeout = float(args.timeout)
        self.client.set_timeout(self.client_timeout)

        self.traffic_manager = self.client.get_trafficmanager(int(args.trafficManagerPort))

        dist = pkg_resources.get_distribution("carla")
        if dist.version != 'leaderboard':
            if LooseVersion(dist.version) < LooseVersion('0.9.10'):
                raise ImportError("CARLA version 0.9.10.1 or newer required. CARLA version found: {}".format(dist))

        # Load agent
        module_name = os.path.basename(args.agent).split('.')[0]
        sys.path.insert(0, os.path.dirname(args.agent))
        self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        self.manager = ScenarioManager(args.timeout, args.routes, args.debug > 1)

        # Time control for summary purposes
        self._start_time = GameTime.get_time()
        self._end_time = None

        # Create the agent timer
        self._agent_watchdog = Watchdog(int(float(args.timeout)))
        #signal.signal(signal.SIGINT, self._signal_handler)

        self.entry_status = ""
        self.crash_message = ""

        self.timestamp = None

        self.static_collisions = []
        self.vehicle_collisions = []
        self.pedestrian_collisions = []

        #self._get_total_potential_collisions()

    '''
    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError("Timeout: Agent took too long to setup")
        elif self.manager:
            self.manager.signal_handler(signum, frame)
    '''

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """

        self._cleanup()
        if hasattr(self, 'manager') and self.manager:
            del self.manager
        if hasattr(self, 'world') and self.world:
            del self.world

    def _cleanup(self):
        """
        Remove and destroy all actors
        """

        # Simulation still running and in synchronous mode?
        if self.manager and self.manager.get_running_status() \
                and hasattr(self, 'world') and self.world:
            # Reset to asynchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(False)

        if self.manager:
            self.manager.cleanup()

        CarlaDataProvider.cleanup()

        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i]:
                self.ego_vehicles[i].destroy()
                self.ego_vehicles[i] = None
        self.ego_vehicles = []

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        if hasattr(self, 'agent_instance') and self.agent_instance:
            self.agent_instance.destroy()
            self.agent_instance = None

        if hasattr(self, 'statistics_manager') and self.statistics_manager:
            self.statistics_manager.scenario = None

        self.entry_status = ""
        self.crash_message = ""

    def _prepare_ego_vehicles(self, ego_vehicles, wait_for_ego_vehicles=False):
        """
        Spawn or update the ego vehicles
        """

        if not wait_for_ego_vehicles:
            for vehicle in ego_vehicles:
                self.ego_vehicles.append(CarlaDataProvider.request_new_actor(vehicle.model,
                                                                             vehicle.transform,
                                                                             vehicle.rolename,
                                                                             color=vehicle.color,
                                                                             vehicle_category=vehicle.category))

        else:
            ego_vehicle_missing = True
            while ego_vehicle_missing:
                self.ego_vehicles = []
                ego_vehicle_missing = False
                for ego_vehicle in ego_vehicles:
                    ego_vehicle_found = False
                    carla_vehicles = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')
                    for carla_vehicle in carla_vehicles:
                        if carla_vehicle.attributes['role_name'] == ego_vehicle.rolename:
                            ego_vehicle_found = True
                            self.ego_vehicles.append(carla_vehicle)
                            break
                    if not ego_vehicle_found:
                        ego_vehicle_missing = True
                        break

            for i, _ in enumerate(self.ego_vehicles):
                self.ego_vehicles[i].set_transform(ego_vehicles[i].transform)

        # sync state
        CarlaDataProvider.get_world().tick()

    def _load_and_wait_for_world(self, args, town, ego_vehicles=None):
        """
        Load a new CARLA world and provide data to CarlaDataProvider
        """

        self.world = self.client.load_world(town)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        self.world.reset_all_traffic_lights()

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(int(args.trafficManagerPort))

        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(int(args.trafficManagerSeed))

        # Wait for the world to be ready
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

        if CarlaDataProvider.get_map().name != town:
            raise Exception("The CARLA server uses the wrong map!"
                            "This scenario requires to use map {}".format(town))

    def _register_statistics(self, config, checkpoint, entry_status, crash_message=""):
        """
        Computes and saved the simulation statistics
        """
        # register statistics
        current_stats_record = self.statistics_manager.compute_route_statistics(
            config,
            self.manager.scenario_duration_system,
            self.manager.scenario_duration_game,
            crash_message
        )

        print("\033[1m> Registering the route statistics\033[0m")
        print('Config index: ', config.index)
        self.statistics_manager.save_record(current_stats_record, config.index, checkpoint)
        self.statistics_manager.save_entry_status(entry_status, False, checkpoint)

    def _load_scenario(self, args, config):
        """
        Load and run the scenario given by config.

        Depending on what code fails, the simulation will either stop the route and
        continue from the next one, or report a crash and stop.
        """
        self.crash_message = ""
        self.entry_status = "Started"

        print("\n\033[1m========= Preparing {} (repetition {}) =========".format(config.name, config.repetition_index))
        print("> Setting up the agent\033[0m")

        # Prepare the statistics of the route
        self.statistics_manager.set_route(config.name, config.index)

        # Set up the user's agent, and the timer to avoid freezing the simulation
        try:
            self._agent_watchdog.start()
            agent_class_name = getattr(self.module_agent, 'get_entry_point')()
            self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config)
            #if not self.agent_instance.initialized:
            #    self.agent_instance._init()
            config.agent = self.agent_instance

            # Check and store the sensors
            if not self.sensors:
                self.sensors = self.agent_instance.sensors()
                track = self.agent_instance.track

                AgentWrapper.validate_sensor_configuration(self.sensors, track, args.track)

                self.sensor_icons = [sensors_to_icons[sensor['type']] for sensor in self.sensors]
                self.statistics_manager.save_sensors(self.sensor_icons, args.checkpoint)

            self._agent_watchdog.stop()

        except SensorConfigurationInvalid as e:
            # The sensors are invalid -> set the ejecution to rejected and stop
            print("\n\033[91mThe sensor's configuration used is invalid:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            self.crash_message = "Agent's sensors were invalid"
            self.entry_status = "Rejected"

            self._register_statistics(config, args.checkpoint, self.entry_status, self.crash_message)
            self._cleanup()
            sys.exit(-1)

        except Exception as e:
            # The agent setup has failed -> start the next route
            print("\n\033[91mCould not set up the required agent:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            self.crash_message = "Agent couldn't be set up"

            self._register_statistics(config, args.checkpoint, self.entry_status, self.crash_message)
            self._cleanup()
            return

        print("\033[1m> Loading the world\033[0m")

        # Load the world and the scenario
        try:
            self._load_and_wait_for_world(args, config.town, config.ego_vehicles)
            self._prepare_ego_vehicles(config.ego_vehicles, False)
            print('making route scenario object')
            scenario = RouteScenario(world=self.world, config=config, debug_mode=args.debug)
            print('done with route scenario object')
            self.statistics_manager.set_scenario(scenario.scenario)
            if args.agent.split('/')[-1] == 'route_following_agent.py':
                self.agent_instance.setup_behavior_agent(CarlaDataProvider.get_map(), scenario.ego_vehicles)

            # self.agent_instance._init()
            # self.agent_instance.sensor_interface = SensorInterface()

            # Night mode
            if config.weather.sun_altitude_angle < 0.0:
                for vehicle in scenario.ego_vehicles:
                    vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))

            # Load scenario and run it
            if args.record:
                self.client.start_recorder("{}/{}_rep{}.log".format(args.record, config.name, config.repetition_index))
            self.manager.load_scenario(scenario, self.agent_instance, config.repetition_index)

        except Exception as e:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            self.crash_message = "Simulation crashed"
            self.entry_status = "Crashed"

            self._register_statistics(config, args.checkpoint, self.entry_status, self.crash_message)

            if args.record:
                self.client.stop_recorder()

            self._cleanup()
            sys.exit(-1)

        print("\033[1m> Running the route\033[0m")

        return scenario

        
        ''' ## starts here
        # Run the scenario
        # try:
        self.manager.run_scenario()
        #self.manager._running = True
        #while self.manager._running:
        #    self.manager.run_scenario_step()



        # except AgentError as e:
        #     # The agent has failed -> stop the route
        #     print("\n\033[91mStopping the route, the agent has crashed:")
        #     print("> {}\033[0m\n".format(e))
        #     traceback.print_exc()

        #     crash_message = "Agent crashed"

        # except Exception as e:
        #     print("\n\033[91mError during the simulation:")
        #     print("> {}\033[0m\n".format(e))
        #     traceback.print_exc()

        #     crash_message = "Simulation crashed"
        #     entry_status = "Crashed"

        # Stop the scenario
        try:
            print("\033[1m> Stopping the route\033[0m")
            self.manager.stop_scenario()
            self._register_statistics(config, args.checkpoint, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            # Remove all actors
            scenario.remove_all_actors()

            self._cleanup()

        except Exception as e:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"

        if crash_message == "Simulation crashed":
            sys.exit(-1)
        ''' ## ends here

    def setup_route_and_scenario(self, args):
        self.route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)
        self.route_indexer.set_route_num(args.route_num)
        print('ROUTE INDEXER: ', len(self.route_indexer))


        
        if args.resume:
            self.route_indexer.resume(args.checkpoint)
            self.statistics_manager.resume(args.checkpoint)
        else:
            self.statistics_manager.clear_record(args.checkpoint)
            self.route_indexer.save_state(args.checkpoint)

        print('RI Current Index: ', self.route_indexer._index)

        self.route_config = self.route_indexer.next()
        self.route_config.index = 0 

        self.scenario = self._load_scenario(args, self.route_config)

        #print('Scenario route: ', self.scenario.route)
        #print('Scenariosssss: ', len(self.scenario.list_scenarios), len(self.scenario.sampled_scenarios_definitions))
        #print('Scenario list: ', self.scenario.list_scenarios)


        return self.route_indexer

    '''
    def get_total_potential_collisions(self):
        vehicle_obstacles = []
        pedestrian_obstacles = []
        static_obstacles = []

        for s in self.scenario.list_scenarios:
            if isinstance(s, srunner.scenarios.object_crash_vehicle.DynamicObjectCrossing):
                if s._adversary_type == True:
                    vehicle_obstacles.append(s)
                else:
                    pedestrian_obstacles.append(s)

            elif isinstance(s, srunner.scenarios.object_crash_vehicle.StationaryObjectCrossing):
                static_obstacles.append(s)

        return static_obstacles, pedestrian_obstacles, vehicle_obstacles
    '''

    def _get_total_potential_collisions(self):
        actors_list = self.world.get_actors()
        self.vehicles = list(actors_list.filter('vehicle.*'))
        #print('Vehicles: ', vehicles)
        self.pedestrians = list(actors_list.filter('walker.*'))
        #print('Pedestrian: ', pedestrians)


    def get_dist_to_closest_actor(self):
        '''
        actors_list = self.world.get_actors()
        vehicles = list(actors_list.filter('vehicle.*'))
        #print('Vehicles: ', vehicles)
        pedestrians = list(actors_list.filter('walker.*'))
        #print('Pedestrian: ', pedestrians)
        '''

        all_collidable_actors = self.vehicles + self.pedestrians
        #print('Actors: ', all_collidable_actors)

        dist_to_closest_actor = []
        
        for v in self.scenario.ego_vehicles:
            ego_location = v.get_location()
            ego_wp = self.world.get_map().get_waypoint(ego_location, project_to_road=True, lane_type=carla.LaneType.Driving)
            #closest_dist = sorted([ego_location.distance(actor.get_location()) for actor in all_collidable_actors] if (actor.get))[1] # the first result in the list is of the ego vehicle itself
            #print('Dists: ', closest_dist)
            #dist_to_closest_actor.append(closest_dist)
            closest_dist = []
            for a in all_collidable_actors:
                a_loc = a.get_location()
                closest_dist.append([ego_location.distance(a_loc), a_loc.x, a_loc.y])
                #if (abs(a_loc.x - ego_wp.transform.location.x) <= 0.5) or (abs(a_loc.y - ego_wp.transform.location.y) <= 0.5):
                    #closest_dist.append([ego_location.distance(a_loc), a_loc.x, a_loc.y])
            #print('Yo: ', closest_dist)
            if len(closest_dist) <= 1:
                dist_to_closest_actor.append(None)
            else:
                dist_to_closest_actor.append(sorted(closest_dist, key=lambda x: x[0])[1:11])

        return dist_to_closest_actor


    '''
    def get_dense_trajectory(self):
        #print('Scenario route: ', self.scenario.route)
        dense_traj = []
        for i, r in enumerate(self.scenario.route):
            print('DT: ', i, r)
            l = r[0].location
            dense_traj.append([l.x, l.y])
        return dense_traj
    '''

    def get_is_at_traffic_light(self):
        tmp = []
        for v in self.scenario.ego_vehicles:
            tmp.append(v.is_at_traffic_light())

        return tmp

    def get_dense_trajectory(self):
        return self.scenario.route

    def get_ego_vehicle_size(self):
        lengths = []
        widths = []

        for ego in self.scenario.ego_vehicles:
            bb_extent = ego.bounding_box.extent
            print('Ego bb: ', bb_extent)
            lengths.append(bb_extent.x * 2)
            widths.append(bb_extent.y * 2)

        return lengths, widths

    def get_ego_acceleration(self):
        acc = []
        for ego in self.scenario.ego_vehicles:
            a = ego.get_acceleration()
            acc.append(math.sqrt(a.x**2 + a.y**2 + a.z**2))

        return acc


    def get_ego_speed(self):
        speed = []
        for ego in self.scenario.ego_vehicles:
            v = ego.get_velocity()
            speed.append(math.sqrt(v.x**2 + v.y**2 + v.z**2))
        return speed

    def get_ego_velocity(self):
        vel = []
        for ego in self.scenario.ego_vehicles:
            v = ego.get_velocity()
            vel.append([v.x, v.y, v.z])
        return vel

    def get_ego_location(self):
        loc = []
        for ego in self.scenario.ego_vehicles:
            l = ego.get_location()
            loc.append([l.x, l.y, l.z])
        return loc

    def get_ego_heading(self):
        import math

        heading = []
        for ego in self.scenario.ego_vehicles:
            v = ego.get_velocity()
            l = ego.get_location()

            #v = np.array([v.x, v.y])
            #v_unit = v / (np.linalg.norm(v) + 0.00001)
            #heading.append(math.atan2(v_unit[1], v_unit[0]))
            heading.append(np.arccos(v.x / (0.001 + np.sqrt(v.x ** 2 + v.y ** 2))))
            #h_vel = math.atan2(v.y, v.x)
            #h_loc = math.atan2(l.y, l.x)
            #print('H: ', h_vel, h_loc)
            #heading.append(h_vel)
        return heading

    def get_scenario_data(self, is_first_reading=False):
        tick_data = self._read_sensors()

        if is_first_reading:
            self.manager.run_scenario_step(tick_data, self.timestamp)
            tick_data = self._read_sensors()

        return tick_data 


    def _read_sensors(self):
        if not self.agent_instance.initialized:
            self.agent_instance._init()

        
        tick_data = None
        
        if self.manager._running:    
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp

            if timestamp:
                self.timestamp = timestamp
                GameTime.on_carla_tick(timestamp)
                CarlaDataProvider.on_carla_tick()
                input_data = self.agent_instance.sensor_interface.get_data()
                tick_data = self.agent_instance.tick(input_data)

        #print('testing: ', tick_data.keys())
        #print('obstacle detect: ', tick_data['obstacle_detect'])


        if tick_data is not None and tick_data.get('rgb', None) is not None:
            tick_data['rgb_preprocessed'] = self.agent_instance.preprocess_image(tick_data['rgb']) 
        
        if tick_data is not None and tick_data.get('lidar', None) is not None:
            tick_data['lidar_preprocessed'] = self.agent_instance.preprocess_lidar(tick_data, tick_data['lidar'])

        '''
        input_data = None
        while input_data is None:
            time.sleep(1)
            input_data = self.agent_instance.sensor_interface.get_data()
            print(input_data)
        tick_data = self.agent_instance.tick(input_data)
        '''

        return tick_data

    def _check_for_collision(self):
        col_s = []
        col_p = []
        col_v = []

        for node in self.statistics_manager._master_scenario.get_criteria():
            if node.list_traffic_events:
                for event in node.list_traffic_events:
                    if event.get_type() == TrafficEventType.COLLISION_STATIC:
                        col_s.append(event.get_dict())
                    elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN:
                        col_p.append(event.get_dict())
                    elif event.get_type() == TrafficEventType.COLLISION_VEHICLE:
                        col_v.append(event.get_dict())

        if len(col_s) > len(self.static_collisions):
            print('Num static collisions: ', col_s, self.static_collisions)
            col_dict = col_s[-1]
            col_dict['ego_location'] = self.get_ego_location()
            col_dict['ego_velocity'] = self.get_ego_velocity()
            self.static_collisions.append(col_dict)

        if len(col_p) > len(self.pedestrian_collisions):
            print('Num pedestrian collisions: ', col_p, self.pedestrian_collisions)
            self.pedestrian_collisions.append(col_p[-1])

        if len(col_v) > len(self.vehicle_collisions):
            print('Num vehicle collisions: ', col_v, self.vehicle_collisions)
            col_dict = col_v[-1]
            col_dict['ego_location'] = self.get_ego_location()
            col_dict['ego_velocity'] = self.get_ego_velocity()
            other_vel = self.world.get_actor(col_dict['id']).get_velocity()
            col_dict['other_velocity'] = [other_vel.x, other_vel.y, other_vel.z]
            self.vehicle_collisions.append(col_dict)

    def run_step(self, tick_data):
        #print('In run step leaderboardeval!!: ', type(tick_data['rgb_preprocessed']))
        self.manager.run_scenario_step(tick_data, self.timestamp)
        self._check_for_collision()

    def stop_scenario(self, args):
        try:
            print("\033[1m> Stopping the route\033[0m")
            results_text = self.manager.stop_scenario()
            self._register_statistics(self.route_config, args.checkpoint, self.entry_status, self.crash_message)

            if args.record:
                self.client.stop_recorder()

            # Remove all actors
            self.scenario.remove_all_actors()

            self._cleanup()

        except Exception as e:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            self.crash_message = "Simulation crashed"

        if self.crash_message == "Simulation crashed":
            sys.exit(-1)

        print("\033[1m> Registering the global statistics\033[0m")
        global_stats_record = self.statistics_manager.compute_global_statistics(self.route_indexer.total)
        StatisticsManager.save_global_record(global_stats_record, self.sensor_icons, self.route_indexer.total, args.checkpoint)

        return results_text


    def run(self, args):
        """
        Run the challenge mode
        """
        # agent_class_name = getattr(self.module_agent, 'get_entry_point')()
        # self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config)

        route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)
        route_indexer.set_route_num(args.route_num)
        print('ROUTE INDEXER: ', len(route_indexer))


        
        if args.resume:
            route_indexer.resume(args.checkpoint)
            self.statistics_manager.resume(args.checkpoint)
        else:
            self.statistics_manager.clear_record(args.checkpoint)
            route_indexer.save_state(args.checkpoint)

        print('RI Current Index: ', route_indexer._index)

        cnt = 0
        while route_indexer.peek():
            # setup
            config = route_indexer.next()

            # run
            #self._load_and_run_scenario(args, config)
            scenario = self._load_scenario(args, config)
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()
            print('Agent instance: ', self.agent_instance.get_data())

            self.manager._running = True

            '''
            while self.manager._running:
                self.manager.run_scenario_step()

            try:
                print("\033[1m> Stopping the route\033[0m")
                self.manager.stop_scenario()
                self._register_statistics(config, args.checkpoint, self.entry_status, self.crash_message)

                if args.record:
                    self.client.stop_recorder()

                # Remove all actors
                scenario.remove_all_actors()

                self._cleanup()

            except Exception as e:
                print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
                print("> {}\033[0m\n".format(e))
                traceback.print_exc()

                self.crash_message = "Simulation crashed"

            if self.crash_message == "Simulation crashed":
                sys.exit(-1)

            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size())
                except:
                    pass

            route_indexer.save_state(args.checkpoint)
            '''


        # save global statistics
        print("\033[1m> Registering the global statistics\033[0m")
        global_stats_record = self.statistics_manager.compute_global_statistics(route_indexer.total)
        StatisticsManager.save_global_record(global_stats_record, self.sensor_icons, route_indexer.total, args.checkpoint)
        
'''

def main():
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='2000', help='TCP port to listen to (default: 2000)')
    parser.add_argument('--trafficManagerPort', default='8000',
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--trafficManagerSeed', default='0',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--debug', type=int, help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', default="600.0",
                        help='Set the CARLA client timeout value in seconds')

    # simulation setup
    parser.add_argument('--routes',
                        help='Name of the route to be executed. Point to the route_xml_file to be executed.',
                        required=True)
    parser.add_argument('--scenarios',
                        help='Name of the scenario annotation file to be mixed with the route.',
                        required=True)
    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate", required=True)
    parser.add_argument("--agent-config", type=str, help="Path to Agent's configuration file", default="")

    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")

    arguments = parser.parse_args()

    statistics_manager = StatisticsManager()

    try:
        leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager)
        leaderboard_evaluator.run(arguments)

    except Exception as e:
        traceback.print_exc()
    finally:
        del leaderboard_evaluator


if __name__ == '__main__':
    main()

'''