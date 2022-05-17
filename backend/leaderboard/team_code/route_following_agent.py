import carla 
#from agents.navigation.local_planner import LocalPlanner
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.local_planner import RoadOption

from leaderboard.autoagents import autonomous_agent as autonomous_agent
#from team_code.base_agent import BaseAgent

#from route_following.basic_agent import BasicAgent
#from route_following.behavior_agent import BehaviorAgent
#from route_following.local_planner import RoadOption

from PIL import Image
from transfuser.data import scale_and_crop_image
import torch
import cv2

from collections import deque

def get_entry_point():
    return 'RouteFollowingAgent'

class RouteFollowingAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.initialized = False
        self.traj_wp_idx = 1 ## start from the second waypoint in the list of major traj wps

    def sensors(self):
        return [
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z':2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 400, 'height': 300, 'fov': 100,
                    'id': 'rgb'
                    }
                ]

    def _init(self):
        self.initialized = True

    """
    def setup_behavior_agent(self, world_map, ego_vehicles):
        print('Map: ', world_map)
        print('Global plan: ', self._global_plan_world_coord)
        plan = [[world_map.get_waypoint(location=p[0].location, project_to_road=True, lane_type=carla.LaneType.Driving), p[1]] for p in self._global_plan_world_coord]
        
        #self.agent = BehaviorAgent(ego_vehicles[0], behavior='normal')
        self.agent = BehaviorAgent(ego_vehicles[0], behavior='cautious')

        '''
        dense_plan = []
        for i in range(len(self._global_plan_world_coord)-1):
            origin = self._global_plan_world_coord[i]
            destination = self._global_plan_world_coord[i+1]
            wps = self.agent._global_planner.trace_route(origin=origin[0].location, destination=destination[0].location)
            wps[0] = (wps[0], origin[1])
            wps[-1] = (wps[-1], destination[1])
            dense_plan += wps

        for i in range(len(dense_plan)):
            if not isinstance(dense_plan[i], tuple):
                dense_plan[i] = (dense_plan[i], RoadOption.VOID)
        '''

        
        #self.agent = BasicAgent(ego_vehicles[0])
        self.agent.set_global_plan(plan, stop_waypoint_creation=True, clean_queue=True)
        
        #self.agent = LocalPlanner(ego_vehicles[0])
        #self.agent.set_global_plan(plan) #, stop_waypoint_creation=True, clean_queue=True)
    """

    """
    def setup_behavior_agent(self, world_map, ego_vehicles):
        self.world_map = world_map
        self.vehicle = ego_vehicles[0]
        self.agent = BehaviorAgent(ego_vehicles[0], ignore_traffic_light=False, behavior='normal')
        self.agent.set_destination(start_location=ego_vehicles[0].get_location(), end_location=self._global_plan_world_coord[self.traj_wp_idx][0].location, clean=True)



    def run_step(self, tick_data=None, timestamp=None, get_pred=False):
        self.agent.update_information()

        print('Hello hello: ', self.traj_wp_idx, len(self._global_plan_world_coord), len(self.agent.get_local_planner().waypoints_queue))
        if len(self.agent.get_local_planner().waypoints_queue) <= 10 and self.traj_wp_idx <= len(self._global_plan_world_coord):
        #if self.vehicle.get_location().distance(self._global_plan_world_coord[self.traj_wp_idx][0].location) < 5.0 and len(self._global_plan_world_coord) >= self.traj_wp_idx:
            self.traj_wp_idx += 1
            target_loc = self._global_plan_world_coord[self.traj_wp_idx][0].location
            target_wp = self.world_map.get_waypoint(location=target_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            self.agent.reroute([target_wp.transform])
            print('Finished rerouting... ', len(self.agent.get_local_planner().waypoints_queue))

        return self.agent.run_step(debug=True)
    """

    def setup_behavior_agent(self, world_map, ego_vehicles):
        #self.agent = BehaviorAgent(ego_vehicles[0], ignore_traffic_light=False, behavior='aggressive')
        self.agent = BasicAgent(ego_vehicles[0])

        dense_plan = []
        for i in range(len(self._global_plan_world_coord)-1):
            c1 = self._global_plan_world_coord[i]
            c2 = self._global_plan_world_coord[i+1]
            #wps = self.agent._global_planner.trace_route(origin=origin[0].location, destination=destination[0].location)
            
            if i == 0:
                l = ego_vehicles[0].get_location()
                wp1 = world_map.get_waypoint(location=l, project_to_road=True, lane_type=carla.LaneType.Driving)
            else:
                wp1 = world_map.get_waypoint(location=c1[0].location, project_to_road=True, lane_type=carla.LaneType.Driving)
            
            wp2 = world_map.get_waypoint(location=c2[0].location, project_to_road=True, lane_type=carla.LaneType.Driving)
            wps = self.agent._trace_route(start_waypoint=wp1, end_waypoint=wp2)
            if i == 0:
                dense_plan.append(wps[0])
                #dense_plan[0] = (dense_plan[0], c1[1])
            #else:
            dense_plan += wps[1:]
            #dense_plan[-1] = (dense_plan[-1], c2[1])
            
        #for i in range(len(dense_plan)):
        #    print('yo yo: ', i, dense_plan[i], type(dense_plan[i]), isinstance(dense_plan[i], tuple))
        #    if not isinstance(dense_plan[i], tuple):
        #        dense_plan[i] = (dense_plan[i], RoadOption.LANEFOLLOW)

        for i in range(1, len(dense_plan)-1):
            if dense_plan[i][1] == RoadOption.LANEFOLLOW:
                dense_plan[i] = (dense_plan[i][0], dense_plan[i-1][1])

        print('Dense plan: ', dense_plan)

        #self.agent._local_planner._waypoints_queue = deque(dense_plan, maxlen=20000)
        self.agent._local_planner.set_global_plan(dense_plan)

    def run_step(self, tick_data=None, timestamp=None, get_pred=False):
        #self.agent.update_information()
        #print('Incoming wp and dir:', self.agent.incoming_waypoint, self.agent.incoming_direction)
        return self.agent.run_step(debug=True)


    def tick(self, input_data):
        rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        return {'rgb': rgb, 'lidar': None}

    def preprocess_image(self, rgb):
        rgb = Image.fromarray(rgb).convert('RGB')
        rgb = scale_and_crop_image(rgb, crop=256)
        rgb = torch.from_numpy(rgb).unsqueeze(0)
        rgb = rgb.to('cuda', dtype=torch.float32)
        return rgb


