import cv2
import numpy as np

def remove_invalid(disp_arr, points, colors ):
	mask = (
		(disp_arr > disp_arr.min()) &
		#(disp_arr < disp_arr.max()) &
		np.all(~np.isnan(points), axis=1) &
		np.all(~np.isinf(points), axis=1) 
	)    
	return points[mask], colors[mask]

def calc_point_cloud(image, disp, q):
	# ~ print(f"Q = {q}")
	# ~ print(f"focal_length = {q[3][2]}")

	points = cv2.reprojectImageTo3D(disp, q).reshape(-1, 3)

	minz = np.amin(points[:,2])
	maxz = np.amax(points[:,2])
	print("Min Z: " + str(minz))
	print("Max Z: " + str(maxz))
	
	#if our image is color or black and white?
	image_dim = image.ndim
	if (image_dim == 2):  # grayscale
		colors = image.reshape(-1, 1)
	elif (image_dim == 3): #color
		colors = image.reshape(-1, 3)
	else:
		print ("Wrong image data")
		exit (0)
	points, colors = remove_invalid(disp.reshape(-1), points, colors)
	return points, colors


def calc_projected_image(points, colors, r, t, k, dist_coeff, width, height):
	xy, cm = project_points(points, colors, r, t, k, dist_coeff, width, height)
	image = np.zeros((height, width, 3), dtype=colors.dtype)
	image[xy[:, 1], xy[:, 0]] = cm
	return image

def project_points(points, colors, r, t, k, dist_coeff, width, height):
	projected, _ = cv2.projectPoints(points, r, t, k, dist_coeff)
	xy = projected.reshape(-1, 2).astype(np.int)
	mask = (
		(0 <= xy[:, 0]) & (xy[:, 0] < width) &
		(0 <= xy[:, 1]) & (xy[:, 1] < height)
	)
	colorsreturn = colors[mask]
	return xy[mask], colorsreturn

def rotate(arr, anglex, anglez):
	return np.array([  # rx
		[1, 0, 0],
		[0, np.cos(anglex), -np.sin(anglex)],
		[0, np.sin(anglex), np.cos(anglex)]
	]).dot(np.array([  # rz
		[np.cos(anglez), 0, np.sin(anglez)],
		[0, 1, 0],
		[-np.sin(anglez), 0, np.cos(anglez)]
	])).dot(arr)