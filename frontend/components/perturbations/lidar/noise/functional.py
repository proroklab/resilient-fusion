import streamlit as st 

import sys
import random
import numpy as np
import pandas as pd

#np.set_printoptions(threshold=sys.maxsize)

import components.perturbations.lidar.utils as lidar_perturbations_utils

def update_perturbation_settings():
	noise_amount = st.session_state['lidar_noise_amount']
	noise_perturb_percent = st.session_state['lidar_perturb_percent']

	if (isinstance(noise_perturb_percent, tuple) and noise_perturb_percent[1] == 0.0) or (isinstance(noise_perturb_percent, float) and noise_perturb_percent == 0.0):
		st.session_state['lidar_perturbation_settings'].pop('noise', None)
		if len(st.session_state['lidar_perturbation_settings']) == 0:
			st.session_state['perturbed_lidar'] = st.session_state['original_lidar']
		return

	st.session_state['lidar_perturbation_settings']['noise'] = {'type': 'random', 'amount': noise_amount, 'percent to perturb': noise_perturb_percent}

def apply_noise(input_point_cloud, settings):
	print('Lidar Noise Settings: ', settings)
	print('Lidar Noise Input Point Cloud: ', input_point_cloud.shape)

	perturbed_point_cloud = add_random_noise(input_point_cloud=input_point_cloud, percent_of_points_to_perturb=settings['percent to perturb'], noise_amount=settings['amount'])
	print('Perturbed lidar: ', perturbed_point_cloud.shape)

	return perturbed_point_cloud

def add_random_noise(input_point_cloud, percent_of_points_to_perturb, noise_amount):
	if isinstance(percent_of_points_to_perturb, tuple):
		percent_of_points_to_perturb = random.uniform(a=percent_of_points_to_perturb[0], b=percent_of_points_to_perturb[1])

	if isinstance(noise_amount, tuple):
		noise_amount = random.uniform(a=noise_amount[0], b=noise_amount[1])

	if percent_of_points_to_perturb == 0.0 or noise_amount == 0.0:
		return input_point_cloud

	x, y, z = input_point_cloud[:, 0].copy(), input_point_cloud[:, 1].copy(), input_point_cloud[:, 2].copy()
	print('Lidar Noise Input x, y, z shape: ', x.shape, y.shape, z.shape)

	r, az, el = lidar_perturbations_utils.cart2sph(x, y, z)
	print('Lidar Noise Input r, a, e shape: ', r.shape, az.shape, el.shape)

	total_points = r.shape[0]
	num_points_to_perturb = int(percent_of_points_to_perturb*total_points)
	print('Num points to perturb: ', num_points_to_perturb)
	indices_to_perturb = np.random.choice(total_points, num_points_to_perturb, replace=False)
	r_to_perturb = r[indices_to_perturb]
	az_to_perturb = az[indices_to_perturb]
	el_to_perturb = el[indices_to_perturb]
	
	perturbed_r = compute_perturbed_r(r_to_perturb, noise_amount)
	#print('PERT R: ', perturbed_r)
	perturbed_x, perturbed_y, perturbed_z = lidar_perturbations_utils.sph2cart(perturbed_r, az_to_perturb, el_to_perturb)
	print('Len of perturbed xyz: ', len(perturbed_x), len(perturbed_y), len(perturbed_z), len(indices_to_perturb))

	for xp, yp, zp, i in zip(perturbed_x, perturbed_y, perturbed_z, indices_to_perturb):
		x[i] = xp
		y[i] = yp 
		z[i] = zp
	
	perturbed_point_cloud = np.dstack((x, y, z))
	perturbed_point_cloud = perturbed_point_cloud.squeeze(axis=0)
	#print('PERT PC: ', perturbed_point_cloud)
	print('Pert PC shape: ', perturbed_point_cloud.shape)

	return perturbed_point_cloud

'''
def cart2sph(x, y, z): ## Transform functions from: https://github.com/numpy/numpy/issues/5228
	hxy = np.hypot(x, y)
	radius = np.hypot(hxy, z)
	elevation = np.arctan2(z, hxy)
	azimuth = np.arctan2(y, x)
	return radius, azimuth, elevation

def sph2cart(radius, azimuth, elevation):
	rcos_theta = radius * np.cos(elevation)
	x = rcos_theta * np.cos(azimuth)
	y = rcos_theta * np.sin(azimuth)
	z = radius * np.sin(elevation)
	return x, y, z
'''

def compute_perturbed_r(r_to_perturb, noise_amount=0.1, percent_of_positive_noise=0.5, noise_weight=0.1):
	perturbed_r = []
	for point in r_to_perturb:
		if random.random() < percent_of_positive_noise:
			perturbed_point = point + noise_weight * noise_amount * point
			#perturbed_point = point + noise_amount * point
		else:
			perturbed_point = point - noise_weight * noise_amount * point
			#perturbed_point = point - noise_amount * point
		perturbed_r.append(perturbed_point)

	return np.array(perturbed_r)

'''
def cart2sph(row): ## Transform functions from: https://github.com/numpy/numpy/issues/5228
	x, y, z = row.x, row.y, row.z
	hxy = np.hypot(x, y)
	r = np.hypot(hxy, z)
	el = np.arctan2(z, hxy)
	az = np.arctan2(y, x)
	return r, az, el

def sph2cart(row):
	r, az, el = row.perturbed_polar_r, row.polar_azimuth, row.polar_elevation
	rcos_theta = r * np.cos(el)
	x = rcos_theta * np.cos(az)
	y = rcos_theta * np.sin(az)
	z = r * np.sin(el)
	return x, y, z

def compute_perturbed_r(row, percent_of_positive_noise, noise_weight, amount_of_noise=None):
	if amount_of_noise is None:
		amount_of_noise = random.random()

	if random.random() < percent_of_positive_noise:
		new_r = row.perturbed_polar_r + noise_weight * amount_of_noise * row.perturbed_polar_r
	else:
		new_r = row.perturbed_polar_r - noise_weight * amount_of_noise * row.perturbed_polar_r

	return new_r

def perturb(df, percent_of_samples_to_perturb):
	rows_to_perturb = df.sample(frac=percent_of_samples_to_perturb)
	rows_to_perturb.perturbed_polar_r = rows_to_perturb.apply(compute_perturbed_r, axis=1, percent_of_positive_noise=0.3, noise_weight=0.01)
	rows_to_perturb['perturbed_x'], rows_to_perturb['perturbed_y'], rows_to_perturb['perturbed_z'] = zip(*rows_to_perturb.apply(sph2cart, axis=1))
	df.update(rows_to_perturb)


def perturb_lidar(input_point_cloud, percent_of_samples_to_perturb=0.1):
	print('Amount to perturb: ', percent_of_samples_to_perturb)
	print('Perturb lidar input: ', input_point_cloud.shape)
	df = pd.DataFrame(data=input_point_cloud, columns=["x", "y", "z"])
	print('Perturb Lidar input DF: ', df.shape)
	#df['z_below_neg2'] = df.z.apply(lambda row: 1 if row<=-2.0 else 0)
	
	df['polar_r'], df['polar_azimuth'], df['polar_elevation'] = zip(*df.apply(cart2sph, axis=1))
	df['perturbed_polar_r'], df['perturbed_x'], df['perturbed_y'], df['perturbed_z'] = df['polar_r'], df['x'], df['y'], df['z']
	perturb(df, percent_of_samples_to_perturb)
	
	res = df[['perturbed_x', 'perturbed_y', 'perturbed_z']].to_numpy()
	#res = df[['x', 'y', 'z']].to_numpy()
	print('Pertubed lidar result np: ', res.shape)

	input_test = input_point_cloud.flatten()
	res_test = res.flatten()

	cnt_noisy = total_cnt = 0
	for i, j in zip(input_test, res_test):
		total_cnt += 1
		if i != j:
			cnt_noisy += 1
			print(i, j)
	print('Testing noisy point cloud: {}/{}'.format(cnt_noisy, total_cnt))

	return res
'''