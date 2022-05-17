import numpy as np

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