import imutils
import time
from datetime import datetime

from GUI.BaseWindow import BaseWindow
from db.DBtools import *
from tools.ImageProccessHelper import stereo_depth_map
from tools.PointCloudHelper import *
from config.config import configValues as cfv
import os


class PointCloudWindow(BaseWindow):
	def __init__(self, themeStyle, TextForApp):
		super().__init__(themeStyle, TextForApp)

		# текст окна
		self.TextForApp = TextForApp

		# Переменая для хранения значений облака точек для расчетов
		self.pointcloud = None

		# Camera settimgs
		self.img_width = cfv.PHOTO_WIDTH
		self.img_height = cfv.PHOTO_HEIGHT

		self.disparity = np.zeros((self.img_width, self.img_height), np.uint8)

		# Loading stereoscopic calibration data
		try:
			npzfile = np.load('./data/calibration_data/{}p/stereo_camera_calibration.npz'.format(self.img_height))
		except:
			print("Camera calibration data not found in cache, file ", './data/calibration_data/{}p/stereo_camera_calibration.npz'.format(self.img_height))
		try:
			npzright = np.load('./data/calibration_data/{}p/camera_calibration_right.npz'.format(self.img_height))
		except:
			print("Camera calibration data not found in cache, file ", './data/calibration_data/{}p/camera_calibration_right.npz'.format(self.img_height))  

		# Вытаскиваем калибровочные параметры
		self.leftMapX = npzfile['leftMapX']
		self.leftMapY = npzfile['leftMapY']
		self.rightMapX = npzfile['rightMapX']
		self.rightMapY = npzfile['rightMapY']
		self.QQ = npzfile['dispartityToDepthMap']
		self.right_K = npzright['camera_matrix']

		# Для руссификации
		self.Project_3dPic = ('Project 3dPic', 'Построить 3D')
		self.Show_3DpointCloud = ('Show 3DpointCloud', 'Показать 3D')

	def run(self,image_to_disp, ifCamPi = True) -> None:

		left_column = [
						[self.sg.Image(filename='', key='Image')],
						[self.sg.Button(self.Project_3dPic[self.language], size=(17,2))],
						[self.sg.Button(self.Show_3DpointCloud[self.language], size=(17,2))]
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


		# FOR PROJECT 3DPiCture
		imgLtoShow = np.zeros((cfv.WINDOW_HEIGHT, cfv.WINDOW_WIDTH), np.uint8)
		projected_image_toshow = np.zeros((cfv.WINDOW_HEIGHT, cfv.WINDOW_WIDTH), np.uint8)
		disparity_to_show   = np.zeros((cfv.WINDOW_HEIGHT, cfv.WINDOW_WIDTH), np.uint8)
		# ------------------------------

		# Основной цикл
		while True:
			event, values = window.read(timeout=200)
			if not event:
				break
			if event == self.sg.WIN_CLOSED or event == 'Cancel':
				break


			if event == self.Project_3dPic[self.language]:
				projected_image_toshow, disparity_to_show, imgLtoShow = self.createCloudPointsInPLY(image_to_disp, ifCamPi)

			if event == self.Show_3DpointCloud[self.language]:
				os.system("meshlab output.ply")


			window.FindElement('Image').Update(data=cv2.imencode('.png', imgLtoShow)[1].tobytes())
			window.FindElement('ProjectedPicture').Update(data=cv2.imencode('.png', projected_image_toshow)[1].tobytes())	
			window.FindElement('DisparityPicture').Update(data=cv2.imencode('.png', disparity_to_show)[1].tobytes())

		window.close()

	# Создает облако точек для отображения и сохраняет его
	def createCloudPointsInPLY(self, image_to_disp, ifCamPi):
		map_width = cfv.IMAGE_WIDTH
		map_height = cfv.IMAGE_HEIGHT

		r = np.eye(3)
		t = np.array([0, 0.0, 100.5])

		pair_img = None
		if ifCamPi:
			pair_img = cv2.cvtColor(image_to_disp, cv2.COLOR_BGR2GRAY)
		else:
			pair_img = cv2.imread(image_to_disp,0)
		
		# Cutting stereopair to the left and right images
		imgLeft = pair_img [0:self.img_height,0:int(self.img_width/2)] #Y+H and X+W
		imgRight = pair_img [0:self.img_height,int(self.img_width/2):self.img_width] #Y+H and X+W
		
		# Undistorting images
		imgL = cv2.remap(imgLeft, self.leftMapX, self.leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
		imgR = cv2.remap(imgRight, self.rightMapX, self.rightMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
		rectified_pair = (imgL, imgR)
		imgLtoShow = cv2.resize (imgL, dsize=(cfv.WINDOW_WIDTH, cfv.WINDOW_HEIGHT), interpolation = cv2.INTER_CUBIC)
		imgRtoShow = cv2.resize (imgR, dsize=(cfv.WINDOW_WIDTH, cfv.WINDOW_HEIGHT), interpolation = cv2.INTER_CUBIC)
			
		# Disparity map calculation
		self.disparity,  native_disparity  = stereo_depth_map(rectified_pair, self.db.mWinParameters)
		disparity_to_show = cv2.resize (self.disparity, dsize=(cfv.WINDOW_WIDTH, cfv.WINDOW_HEIGHT), interpolation = cv2.INTER_CUBIC)

		# Point cloud calculation   
		points_3, colors = calc_point_cloud(self.disparity, native_disparity, self.QQ)

		# Camera settings for the pointcloud projection 
		dist_coeff = np.zeros((4, 1))

		# Сохраняем облако точек в файл
		self.__write_ply("output.ply", points_3, colors)
		
		projected_image = calc_projected_image(points_3, colors, r, t, self.right_K, dist_coeff, map_width, map_height)
		projected_image_toshow = cv2.resize (projected_image, dsize=(cfv.WINDOW_WIDTH, cfv.WINDOW_HEIGHT), interpolation = cv2.INTER_CUBIC)

		return projected_image_toshow, disparity_to_show, imgLtoShow

	def updatePointCloud(self, image_to_disp, ifCamPi) -> None:

		pair_img = None
		if ifCamPi:
			pair_img = cv2.cvtColor(image_to_disp, cv2.COLOR_BGR2GRAY)
		else:
			pair_img = cv2.imread(image_to_disp,0)

		# Cutting stereopair to the left and right images
		imgLeft = pair_img [0:self.img_height,0:int(self.img_width/2)] #Y+H and X+W
		imgRight = pair_img [0:self.img_height,int(self.img_width/2):self.img_width] #Y+H and X+W
		
		# Undistorting images
		imgL = cv2.remap(imgLeft, self.leftMapX, self.leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
		imgR = cv2.remap(imgRight, self.rightMapX, self.rightMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
		rectified_pair = (imgL, imgR)

		_,  native_disparity  = stereo_depth_map(rectified_pair, self.db.mWinParameters)

		self.pointcloud = cv2.reprojectImageTo3D(native_disparity, self.QQ)

	# Функция для сохранения облака точек 
	@staticmethod
	def __write_ply(fn, verts, colors) -> None:
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

		verts = verts.reshape(-1, 3)
		colors = colors.reshape(-1, 3)
		verts = np.hstack([verts, colors])
		with open(fn, 'wb') as f:
			f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
			np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
		print ("Pointcloud saved to "+fn)