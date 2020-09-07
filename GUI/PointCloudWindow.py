import cv2
import numpy as np
import imutils
import time
from datetime import datetime

from GUI.baseWindow import BaseWindow
from db.DBtools import *
# from point_cloud import *
from tools.determine_object_helper import stereo_depth_map


class pointCloudWindow(BaseWindow):
	def __init__(self, themeStyle, TextForApp):
		super().__init__(themeStyle, TextForApp)

		self.TextForApp = TextForApp

	def run(self,image_to_disp, parameters, ifCamPi = True):

		left_column = [
						[self.sg.Image(filename='', key='Image')],
						[self.sg.Button('Project 3dPic', size=(17,2))],
						[self.sg.Button('build 3DpointCloud', size=(17,2))]
		]

		right_column = [
					[self.sg.Image(filename='', key='ProjectedPicture')],
					[self.sg.Image(filename='', key='DisparityPicture')]
		]

		layout  = [[self.sg.Column(left_column, element_justification='c'), 
					self.sg.VSeperator(),
					self.sg.Column(right_column, element_justification='c')]
					]

		# Наше окно
		window = self.sg.Window(self.TextForApp, 
						layout, icon=self.icon, resizable=True)


		# ------------------------------FOR PROJECT 3DPICture
		imgLtoShow = np.zeros((362, 640), np.uint8)
		projected_image_toshow = np.zeros((362, 640), np.uint8)
		disparity_to_show   = np.zeros((362, 640), np.uint8)


		# ------------------------------

		while True:
			event, values = window.read(timeout=200)
			if not event:
				break
			if event == self.sg.WIN_CLOSED or event == 'Cancel':
				break



			if event == 'Project 3dPic':
				_, projected_image_toshow, disparity_to_show, imgLtoShow = self.createCloudPoints(image_to_disp, parameters, ifCamPi)

			if event == 'build 3DpointCloud':
				pass


			window.FindElement('Image').Update(data=cv2.imencode('.png', imgLtoShow)[1].tobytes())
			window.FindElement('ProjectedPicture').Update(data=cv2.imencode('.png', projected_image_toshow)[1].tobytes())	
			window.FindElement('DisparityPicture').Update(data=cv2.imencode('.png', disparity_to_show)[1].tobytes())

		window.close()

	def createCloudPoints(self, image_to_disp, parameters, ifCamPi):
		

		ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar blue
		property uchar green
		property uchar red
		end_header
		'''


		print ("You can press 'Q' to quit this script!")
		time.sleep (5)


		# Camera settimgs
		img_width = 2560
		img_height = 720

		# Image for disparity
		#imageToDisp = './scenes4test/scene_2560x720_3.png'

		disparity = np.zeros((img_width, img_height), np.uint8)


		# Loading stereoscopic calibration data
		try:
			npzfile = np.load('./data/calibration_data/{}p/stereo_camera_calibration.npz'.format(img_height))
		except:
			print("Camera calibration data not found in cache, file ", './data/calibration_data/{}p/stereo_camera_calibration.npz'.format(img_height))
		try:
			npzright = np.load('./data/calibration_data/{}p/camera_calibration_right.npz'.format(img_height))
		except:
			print("Camera calibration data not found in cache, file ", './data/calibration_data/{}p/camera_calibration_right.npz'.format(img_height))  
			
			
		imageSize = tuple(npzfile['imageSize'])
		leftMapX = npzfile['leftMapX']
		leftMapY = npzfile['leftMapY']
		rightMapX = npzfile['rightMapX']
		rightMapY = npzfile['rightMapY']
		QQ = npzfile['dispartityToDepthMap']
		right_K = npzright['camera_matrix']
		#print (right_K)
		#print (QQ)
		#exit(0)

		map_width = 1280
		map_height = 720

		min_y = 10000
		max_y = -10000
		min_x =  10000
		max_x = -10000
		min_z =  10000
		max_z = -10000

		def remove_invalid(disp_arr, points, colors ):
			mask = (
				(disp_arr > disp_arr.min()) &
				#(disp_arr < disp_arr.max()) &
				np.all(~np.isnan(points), axis=1) &
				np.all(~np.isinf(points), axis=1) 
			)    
			return points[mask], colors[mask]

		def calc_point_cloud(image, disp, q):
			print(f"Q = {q}")
			print(f"focal_length = {q[3][2]}")

			points = cv2.reprojectImageTo3D(disp, q).reshape(-1, 3)
			print(f"shape of points = {points.shape}")

			points2 = cv2.reprojectImageTo3D(disp, q)
			print(f"shape of points2 = {points2.shape}")
			print(f"A1 = {points2[int(290*720/362)][int(199*1280/640)]}")
			print(f"B1 = {points2[int(267*720/362)][int(515*1280/640)]}")

			print(f"A2 = {points2[int(303*720/362)][int(197*1280/640)]}")
			print(f"B2 = {points2[int(289*720/362)][int(510*1280/640)]}")


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
			return points2, points, colors

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

		angles = {  # x, z
			'w': (-np.pi/40, 0),
			's': (np.pi/40, 0),
			'a': (0, np.pi/40),
			'd': (0, -np.pi/40)
			}
		r = np.eye(3)
		t = np.array([0, 0.0, 100.5])

		def write_ply(fn, verts, colors):
			verts = verts.reshape(-1, 3)
			colors = colors.reshape(-1, 3)
			verts = np.hstack([verts, colors])
			with open(fn, 'wb') as f:
				f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
				np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
			print ("Pointcloud saved to "+fn)


		pair_img = None
		#pair_img = cv2.imread(image_to_disp,0)
		if ifCamPi:
			pair_img = cv2.cvtColor(image_to_disp, cv2.COLOR_BGR2GRAY)
		else:
			pair_img = cv2.imread(image_to_disp,0)
		
		# Cutting stereopair to the left and right images
		imgLeft = pair_img [0:img_height,0:int(img_width/2)] #Y+H and X+W
		imgRight = pair_img [0:img_height,int(img_width/2):img_width] #Y+H and X+W
		
		# Undistorting images
		imgL = cv2.remap(imgLeft, leftMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
		imgR = cv2.remap(imgRight, rightMapX, rightMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
		rectified_pair = (imgL, imgR)
		imgLtoShow = cv2.resize (imgL, dsize=(640, 362), interpolation = cv2.INTER_CUBIC)
		imgRtoShow = cv2.resize (imgR, dsize=(640, 362), interpolation = cv2.INTER_CUBIC)
			
		# Disparity map calculation
		disparity,  native_disparity  = stereo_depth_map(rectified_pair, parameters)
		disparity_to_show = cv2.resize (disparity, dsize=(640, 362), interpolation = cv2.INTER_CUBIC)

		# Point cloud calculation   
		points2, points_3, colors = calc_point_cloud(disparity, native_disparity, QQ)
		#print(f"points_3 == {points_3}")

		# Camera settings for the pointcloud projection 
		k = right_K
		dist_coeff = np.zeros((4, 1))
			
		t2 = datetime.now()

		write_ply("output.ply", points_3, colors)
		
		projected_image = calc_projected_image(points_3, colors, r, t, k, dist_coeff, map_width, map_height)
		projected_image_toshow = cv2.resize (projected_image, dsize=(640, 362), interpolation = cv2.INTER_CUBIC)

		return points2, projected_image_toshow, disparity_to_show, imgLtoShow

	def forFormula(self, image_to_disp, parameters, ifCamPi):

		ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar blue
		property uchar green
		property uchar red
		end_header
		'''


		# Camera settimgs
		img_width = 2560
		img_height = 720

		# Image for disparity
		#imageToDisp = './scenes4test/scene_2560x720_3.png'

		disparity = np.zeros((img_width, img_height), np.uint8)


		# Loading stereoscopic calibration data
		try:
			npzfile = np.load('./data/calibration_data/{}p/stereo_camera_calibration.npz'.format(img_height))
		except:
			print("Camera calibration data not found in cache, file ", './data/calibration_data/{}p/stereo_camera_calibration.npz'.format(img_height))
		try:
			npzright = np.load('./data/calibration_data/{}p/camera_calibration_right.npz'.format(img_height))
		except:
			print("Camera calibration data not found in cache, file ", './data/calibration_data/{}p/camera_calibration_right.npz'.format(img_height))  
			
			
		imageSize = tuple(npzfile['imageSize'])
		leftMapX = npzfile['leftMapX']
		leftMapY = npzfile['leftMapY']
		rightMapX = npzfile['rightMapX']
		rightMapY = npzfile['rightMapY']
		QQ = npzfile['dispartityToDepthMap']
		right_K = npzright['camera_matrix']
		#print (right_K)
		#print (QQ)
		#exit(0)

		map_width = 1280
		map_height = 720

		min_y = 10000
		max_y = -10000
		min_x =  10000
		max_x = -10000
		min_z =  10000
		max_z = -10000

		pair_img = None
		#pair_img = cv2.imread(image_to_disp,0)
		if ifCamPi:
			pair_img = cv2.cvtColor(image_to_disp, cv2.COLOR_BGR2GRAY)
		else:
			pair_img = cv2.imread(image_to_disp,0)

		# Cutting stereopair to the left and right images
		imgLeft = pair_img [0:img_height,0:int(img_width/2)] #Y+H and X+W
		imgRight = pair_img [0:img_height,int(img_width/2):img_width] #Y+H and X+W
		
		# Undistorting images
		imgL = cv2.remap(imgLeft, leftMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
		imgR = cv2.remap(imgRight, rightMapX, rightMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
		rectified_pair = (imgL, imgR)

		disparity,  native_disparity  = stereo_depth_map(rectified_pair, parameters)

		points2 = cv2.reprojectImageTo3D(native_disparity, QQ)

		return points2

