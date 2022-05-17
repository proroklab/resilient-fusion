import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque

import torch
#torch.set_printoptions(profile="full")

import carla
import numpy as np
from PIL import Image

from leaderboard.autoagents import autonomous_agent
from transfuser.model import TransFuser
from transfuser.config import GlobalConfig
from transfuser.data import scale_and_crop_image, lidar_to_histogram_features, transform_2d_points
from team_code.planner import RoutePlanner

import math
from matplotlib import cm

SAVE_PATH = os.environ.get('SAVE_PATH', None)


def get_entry_point():
	return 'TransFuserAgent'


class TransFuserAgent(autonomous_agent.AutonomousAgent):
	def setup(self, path_to_conf_file):
		self.lidar = None
		self.lidar_processed = list()
		self.track = autonomous_agent.Track.SENSORS
		self.config_path = path_to_conf_file
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		self.input_buffer = {'rgb': deque(), 'rgb_left': deque(), 'rgb_right': deque(), 
							'rgb_rear': deque(), 'lidar': deque(), 'gps': deque(), 'thetas': deque()}

		self.input_buffer_pert = {'rgb': deque(), 'rgb_left': deque(), 'rgb_right': deque(), 
							'rgb_rear': deque(), 'lidar': deque(), 'gps': deque(), 'thetas': deque()}

		self.config = GlobalConfig()
		self.net = TransFuser(self.config, 'cuda')
		self.net.load_state_dict(torch.load(os.path.join(path_to_conf_file, 'best_model.pth')))
		self.net.cuda()
		self.net.eval()

		self.save_path = None
		if SAVE_PATH is not None:
			now = datetime.datetime.now()
			string = pathlib.Path(os.environ['ROUTES']).stem + '_'
			string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

			print (string)

			self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
			self.save_path.mkdir(parents=True, exist_ok=False)

			(self.save_path / 'rgb').mkdir(parents=True, exist_ok=False)
			(self.save_path / 'lidar_0').mkdir(parents=True, exist_ok=False)
			(self.save_path / 'lidar_1').mkdir(parents=True, exist_ok=False)
			(self.save_path / 'meta').mkdir(parents=True, exist_ok=False)


	def _init(self):
		self._route_planner = RoutePlanner(4.0, 50.0)
		self._route_planner.set_route(self._global_plan, True)

		self.initialized = True

	def _get_position(self, tick_data):
		gps = tick_data['gps']
		gps = (gps - self._route_planner.mean) * self._route_planner.scale

		return gps

	def sensors(self):
		return [
				{
					'type': 'sensor.camera.rgb',
					'x': 1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 400, 'height': 300, 'fov': 100,
					'id': 'rgb'
					},
				{
					'type': 'sensor.camera.rgb',
					'x': 1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
					'width': 400, 'height': 300, 'fov': 100,
					'id': 'rgb_left'
					},
				{
					'type': 'sensor.camera.rgb',
					'x': 1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
					'width': 400, 'height': 300, 'fov': 100,
					'id': 'rgb_right'
					},
				{
					'type': 'sensor.camera.rgb',
					'x': -1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': -180.0,
					'width': 400, 'height': 300, 'fov': 100,
					'id': 'rgb_rear'
					},
                {   
                    'type': 'sensor.lidar.ray_cast',
                    'x': 1.3, 'y': 0.0, 'z': 2.5,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                    'id': 'lidar',
                    },
				{
					'type': 'sensor.other.imu',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.05,
					'id': 'imu'
					},
				{
					'type': 'sensor.other.gnss',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.01,
					'id': 'gps'
					},
				{
					'type': 'sensor.speedometer',
					'reading_frequency': 20,
					'id': 'speed'
					}
				]

	def tick(self, input_data):
		self.step += 1

		rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		rgb_rear = cv2.cvtColor(input_data['rgb_rear'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		gps = input_data['gps'][1][:2]
		speed = input_data['speed'][1]['speed']
		compass = input_data['imu'][1][-1]
		if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
			compass = 0.0

		#if self.step == 0 or self.step % 2 != 0:
		#	self.lidar = input_data['lidar'][1][:, :3]
		#lidar = self.lidar
		lidar = input_data['lidar'][1][:, :3]
		print('In tick data: ', self.step, np.amax(lidar), np.amin(lidar))

		result = {
				'rgb': rgb,
				'rgb_left': rgb_left,
				'rgb_right': rgb_right,
				'rgb_rear': rgb_rear,
				'lidar': lidar,
				'gps': gps,
				'speed': speed,
				'compass': compass,
				}
		
		pos = self._get_position(result)
		result['gps'] = pos
		next_wp, next_cmd = self._route_planner.run_step(pos)
		result['next_command'] = next_cmd.value

		theta = compass + np.pi/2
		R = np.array([
			[np.cos(theta), -np.sin(theta)],
			[np.sin(theta), np.cos(theta)]
			])

		local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])
		local_command_point = R.T.dot(local_command_point)
		result['target_point'] = tuple(local_command_point)

		return result

	def get_image_encoder_model(self):
		return self.net.encoder.image_encoder

	def get_lidar_encoder_model(self):
		return self.net.encoder.lidar_encoder

	def preprocess_image(self, rgb):
		rgb = Image.fromarray(rgb).convert('RGB')
		rgb = scale_and_crop_image(rgb, crop=self.config.input_resolution)
		rgb = torch.from_numpy(rgb).unsqueeze(0)
		rgb = rgb.to('cuda', dtype=torch.float32)
		return rgb

	def preprocess_lidar(self, tick_data, lidar, type='original'):
		print('Lidar: ', lidar)
		
		buff = self.input_buffer.copy()
		print('Preprocess test: ', len(self.input_buffer['lidar']), len(self.input_buffer['gps']), len(self.input_buffer['thetas']))
		buff['lidar'].append(lidar)
		buff['gps'].append(tick_data['gps'])
		buff['thetas'].append(tick_data['compass'])
		print('Preprocess test 2: ', len(buff['lidar']), len(buff['gps']), len(buff['thetas']))
		print('Preprocess test values: ', buff['gps'], buff['thetas'])
		
		# transform the lidar point clouds to local coordinate frame
		ego_theta = buff['thetas'][-1]
		ego_x, ego_y = buff['gps'][-1]

		#Only predict every second step because we only get a LiDAR every second frame.
		for i, lidar_point_cloud in enumerate(buff['lidar']):
			print('Lidar pc original', i, np.amax(lidar_point_cloud), np.amin(lidar_point_cloud))
			curr_theta = buff['thetas'][i]
			curr_x, curr_y = buff['gps'][i]
			lidar_point_cloud[:,1] *= -1 # inverts x, y
			lidar_transformed = transform_2d_points(lidar_point_cloud,
						np.pi/2-curr_theta, -curr_x, -curr_y, np.pi/2-ego_theta, -ego_x, -ego_y)
			print('Lidar transformed pre: ', i, np.amax(lidar_transformed), np.amin(lidar_transformed))
			lidar_hist = lidar_to_histogram_features(lidar_transformed, crop=self.config.input_resolution)
			print('Lidar hist: ', i, np.amax(lidar_hist), np.amin(lidar_hist))
			lidar_res = torch.from_numpy(lidar_hist).unsqueeze(0).to('cuda', dtype=torch.float32)
			print('Lidar transformed: ', i, torch.max(lidar_res), torch.min(lidar_res))
			

		buff['lidar'].popleft()
		buff['gps'].popleft()
		buff['thetas'].popleft()

		#self.input_buffer['lidar'].popleft()
		#self.input_buffer['gps'].popleft()
		#self.input_buffer['thetas'].popleft()
		#print('Preprocess test values after pop: ', self.input_buffer['gps'], self.input_buffer['thetas'])

		#return lidar_processed[0]
		return lidar_res


	@torch.no_grad()
	def get_image_features(self, encoder, rgb, lidar=None):
		return encoder(rgb)

	@torch.no_grad()
	def get_lidar_features(self, encoder, lidar):
		return encoder(lidar)

	@torch.no_grad()
	def run_step(self, tick_data, timestamp=None, get_pred=False):
		print('TICK DATA:', tick_data.keys())
		yo = self.preprocess_image(tick_data['rgb']) - tick_data['rgb_preprocessed']
		print('tick rgb yo yo yo!!: ', self.step, torch.max(yo), torch.min(yo))
		#print('step yo yo yo!!: ', self.step)

		if not self.initialized:
			self._init()

		if self.step < self.config.seq_len:
			self.input_buffer['rgb'].append(tick_data['rgb_preprocessed'])
			
			if not self.config.ignore_sides:
				rgb_left = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_left']), crop=self.config.input_resolution)).unsqueeze(0)
				self.input_buffer['rgb_left'].append(rgb_left.to('cuda', dtype=torch.float32))
				
				rgb_right = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_right']), crop=self.config.input_resolution)).unsqueeze(0)
				self.input_buffer['rgb_right'].append(rgb_right.to('cuda', dtype=torch.float32))

			if not self.config.ignore_rear:
				rgb_rear = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_rear']), crop=self.config.input_resolution)).unsqueeze(0)
				self.input_buffer['rgb_rear'].append(rgb_rear.to('cuda', dtype=torch.float32))

			self.input_buffer['lidar'].append(tick_data['lidar'])
			self.input_buffer['gps'].append(tick_data['gps'])
			self.input_buffer['thetas'].append(tick_data['compass'])

			control = carla.VehicleControl()
			control.steer = 0.0
			control.throttle = 0.0
			control.brake = 0.0
			
			return control

		gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)
		command = torch.FloatTensor([tick_data['next_command']]).to('cuda', dtype=torch.float32)

		tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
											torch.FloatTensor([tick_data['target_point'][1]])]
		target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)

		encoding = []
		self.input_buffer['rgb'].popleft()
		self.input_buffer['rgb'].append(tick_data['rgb_preprocessed'])
		
		if not self.config.ignore_sides:
			rgb_left = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_left']), crop=self.config.input_resolution)).unsqueeze(0)
			self.input_buffer['rgb_left'].popleft()
			self.input_buffer['rgb_left'].append(rgb_left.to('cuda', dtype=torch.float32))
			
			rgb_right = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_right']), crop=self.config.input_resolution)).unsqueeze(0)
			self.input_buffer['rgb_right'].popleft()
			self.input_buffer['rgb_right'].append(rgb_right.to('cuda', dtype=torch.float32))

		if not self.config.ignore_rear:
			rgb_rear = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_rear']), crop=self.config.input_resolution)).unsqueeze(0)
			self.input_buffer['rgb_rear'].popleft()
			self.input_buffer['rgb_rear'].append(rgb_rear.to('cuda', dtype=torch.float32))

		#self.input_buffer['lidar'].popleft()
		#self.input_buffer['lidar'].append(tick_data['lidar'])
		#self.input_buffer['gps'].popleft()
		#self.input_buffer['gps'].append(tick_data['gps'])
		#self.input_buffer['thetas'].popleft()
		#self.input_buffer['thetas'].append(tick_data['compass'])

		
		if (self.step  % 2 != 0 or self.step <= 4):

			print('In pred_wp calc function!!', self.step)
			self.pred_wp = self.net(self.input_buffer['rgb'] + self.input_buffer['rgb_left'] + \
							   self.input_buffer['rgb_right']+self.input_buffer['rgb_rear'], \
							   [tick_data['lidar_preprocessed']], target_point, gt_velocity)
		

		#print('In pred_wp calc function!!')
		#self.pred_wp = self.net(self.input_buffer['rgb'] + self.input_buffer['rgb_left'] + \
		#					   self.input_buffer['rgb_right']+self.input_buffer['rgb_rear'], \
		#					   [tick_data['lidar_preprocessed']], target_point, gt_velocity)
		
		steer, throttle, brake, metadata = self.net.control_pid(self.pred_wp, gt_velocity)
		self.pid_metadata = metadata

		if brake < 0.05: brake = 0.0
		if throttle > brake: brake = 0.0

		control = carla.VehicleControl()
		control.steer = float(steer)
		control.throttle = float(throttle)
		control.brake = float(brake)

		if SAVE_PATH is not None and self.step % 10 == 0:
			self.save(tick_data)

		return control

	def save(self, tick_data):
		frame = self.step // 10

		Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%04d.png' % frame))

		Image.fromarray(cm.gist_earth(self.lidar_processed[0].cpu().numpy()[0, 0], bytes=True)).save(self.save_path / 'lidar_0' / ('%04d.png' % frame))
		Image.fromarray(cm.gist_earth(self.lidar_processed[0].cpu().numpy()[0, 1], bytes=True)).save(self.save_path / 'lidar_1' / ('%04d.png' % frame))


		outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
		json.dump(self.pid_metadata, outfile, indent=4)
		outfile.close()

	def destroy(self):
		del self.net

