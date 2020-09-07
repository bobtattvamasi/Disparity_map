import cv2, PySimpleGUI as sg
import base64
import glob
import sys, traceback, os
import subprocess
import math

import numpy as np
import imutils

# Imports from files
from GUI.baseWindow import BaseWindow
from config.config import configValues
from point_cloud import create_points_cloud
from tools.determine_object_helper import convert_to_bytes, calibrate_two_images, stereo_depth_map, contours_finder, resize_rectified_pair, autoFindRect
from db.DBtools import *
from GUI.autoDetectRectWindow import autoDetectRectWindow
from GUI.PointCloudWindow import pointCloudWindow
#from FPS_test import PiVideoStream


  

# Основное окно
class MainWindow(BaseWindow):

	# Функия считывает все картинки из папки
	@staticmethod
	def image_to_disp(path_to_imagesFolder):
		photo_files = []
		i = 0
		for file in glob.glob(path_to_imagesFolder):
			photo_files.append(file)
		return photo_files, i

	def __init__(self, themeStyle, TextForApp, ifCamPi=True):
		super().__init__(themeStyle, TextForApp)
		self.ifCamPi = ifCamPi

		# Camera settimgs
		self.cam_width = configValues.PHOTO_WIDTH
		self.cam_height = configValues.PHOTO_HEIGHT

		if ifCamPi:
			import picamera
			from picamera import PiCamera
				
		
			# Final image capture settings
			scale_ratio = 1
			# # Camera resolution height must be dividable by 16, and width by 32
			cam_width = int((self.cam_width+31)/32)*32
			cam_height = int((self.cam_height+15)/16)*16
			print ("Camera resolution: "+str(cam_width)+" x "+str(cam_height))

			# # Buffer for captured image settings
			self.img_width = int (cam_width * scale_ratio)
			self.img_height = int (cam_height * scale_ratio)
			self.capture = np.zeros((self.img_height, self.img_width, 4), dtype=np.uint8)
			print ("Scaled image resolution: "+str(self.img_width)+" x "+str(self.img_height))

			# # Initialize the camera
			self.camera = PiCamera(stereo_mode='side-by-side',stereo_decimate=False)
			self.camera.resolution=(cam_width, cam_height)
			self.camera.framerate = 20
			self.camera.hflip = True
			#self.vs = PiVideoStream(resolution=(self.cam_width,self.cam_height)).start()
		else:
			# Список имен изображений, индекс-указатель на определенное
			# изображение
			self.imageToDisp, self.index = self.image_to_disp(configValues.folderTestScenes)

		# Считываем значения параметров 
		# из файла .csv
		self.parameters = read_csv("db/settings.csv")

		self.secondWindow = autoDetectRectWindow("DarkAmber", 'Auto Detecting Rectangles')
		self.Window3D = pointCloudWindow("DarkAmber", "3D Point Cloud Builder")

		# Переменная для хранения линий
		# с автоопределения прямоугольников
		self.auto_lines = []

		# Словарь параметров для настройки изображений, 
		# которые используются для автоопределения граней
		# прямоугольных  объектов.
		self.secondWin_parameters = self.secondWindow.secondWin_parameters
		self.disparity_value = None
		self.pointcloud = None

		
		# --------------------------------------------
		# Разделяем пространство окна на левое и правое
		#
		# Левая колонка:
		left_column = [
			# Место под левое изображение
			[self.sg.Image(filename='', key='image')],
			# Далее идут слайдеры параметров для настройки выходной карты глубин
			[self.sg.Frame('Settings',[
			[self.sg.Text("SpklWinSze "), self.sg.Slider(range=(0, 300), orientation='h', size=(34, 10), default_value=self.parameters["SpklWinSze"])],
			[self.sg.Text("SpcklRng   "), self.sg.Slider(range=(0, 40), orientation='h', size=(34, 10), default_value=self.parameters["SpcklRng"])],
			[self.sg.Text("UnicRatio  "), self.sg.Slider(range=(1, 80), orientation='h', size=(34, 10), default_value=self.parameters["UnicRatio"])],
			[self.sg.Text("TxtrThrshld"), self.sg.Slider(range=(0, 1000), orientation='h', size=(34, 10), default_value=self.parameters["TxtrThrshld"])],
			[self.sg.Text("NumOfDisp  "), self.sg.Slider(range=(16, 256), orientation='h', size=(34, 10), default_value=self.parameters["NumOfDisp"])],
			[self.sg.Text("MinDISP    "), self.sg.Slider(range=(-300, 300), orientation='h', size=(34, 10), default_value=self.parameters["MinDISP"])],
			[self.sg.Text("PreFiltCap "), self.sg.Slider(range=(5, 63), orientation='h', size=(34, 10), default_value=self.parameters["PreFiltCap"])],
			[self.sg.Text("PFS        "), self.sg.Slider(range=(5, 255), orientation='h', size=(34, 10), default_value=self.parameters["PFS"])],
			[self.sg.Button("save settings", size=(10,1))]
			])],
			# button for create map of disparity
			[self.sg.Button('create map', size=(15,2)),
			# Кнопка по которой вычисляются размеры найденных граней
			self.sg.Button('find distances', size=(15,2)),
			# Кнопка котороя delete lines
			self.sg.Button('clear lines', size=(15,2))
			]
		]

		if not self.ifCamPi:
			left_column.append([self.sg.Button("Next Picture ->", size=(15,2))])

		# Правая колонка:
		#
		right_column = [
			# Место для отображения второй картинки
			# Только в виде графа, чтобы можно было 
			# рисовать поверх грани, которые мы будем
			# измерять.
			[self.sg.Graph(canvas_size=(640, 362),
					graph_bottom_left=(0, 0),
					graph_top_right=(640, 362),
					key="graph",
					change_submits=True, # mouse click events
					drag_submits=True # mouse drag events]
					)],
			# Вывод всей важной информации происходит здесь
			[self.sg.Output(size=(64, 21), key = '_output_')],
			[self.sg.Button('lineFinder Settings', size=(15,2)), self.sg.Button('auto-lineFinder', size=(15,2))],
			[self.sg.Button('3DCloud Settings', size=(15,2)), self.sg.Button('Show 3D-Points', size=(15,2))]
			
		]

		# Объединяемвсе в один объект
		layout  = [[self.sg.Column(left_column, element_justification='c'), 
					self.sg.VSeperator(),
					self.sg.Column(right_column, element_justification='c')]
				]
		# Наше окно
		self.window = self.sg.Window(TextForApp, 
						layout, icon=self.icon, resizable=True)

		# Variables for main_loop func
		# Флаг для перерисовывания графа(для возможности рисовать на нем)
		self.a_id = None
		# Сам граф
		self.graph = self.window.Element("graph")

		# Флаг и переменные для начала и конца рисования линии
		self.dragging = False
		self.start_point, self.end_point = None, None
		self.start_p, self.end_p = None, None
		self.lines =[]
		self.lines_for_view = []
		# переменные для хранения изображений карт несоответсвий
		self.disparity_to_show = np.zeros((362, 640), np.uint8)
		self.disparity = np.zeros((720,1280), np.uint8)

		
		
	def lineFinder(self, image):
		hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		self.secondWindow.auto_lines = []
		_, mask_frame, thresh = autoFindRect(image, hsv_frame, self, True)
			
		return image


	# Основная функция в котором работает наше окно
	def run(self, ifDebugVersion=False):
		def main_loop(frame):
			event, values = self.window.Read(timeout=20, timeout_key='timeout')      # get events for the window with 20ms max wait

			if event is None or event == self.sg.WIN_CLOSED or event == 'Cancel':  
				return False  

			#imageToDisp = self.vs.read()
			
			imageToDisp = frame

			imgL, imgR = calibrate_two_images(imageToDisp, self.ifCamPi)
			rectified_pair = (imgR, imgL)
			
			try:
				if ifDebugVersion:
					self.disparity, value_disparity = self.deepMap_updater(imageToDisp, values)
				
				if event == 'auto-lineFinder':
					self.window.FindElement("_output_").Update('')
					try:
						self.disparity, self.disparity_value = self.deepMap_updater(imageToDisp, values)
						self.pointcloud = self.Window3D.forFormula(imageToDisp, self.parameters, self.ifCamPi)
						self.disparity = self.lineFinder(self.disparity)
					except:
						print("find lines dont work")
						print(traceback.format_exc())
				
				#
				# Рисовательная часть(возможно не нужно)
				#--------------------------------------
				# Перерисовка графа
				if self.a_id:
					self.graph.DeleteFigure(self.a_id)
				self.disparity_to_show = cv2.resize(self.disparity, dsize=(640, 362), interpolation = cv2.INTER_CUBIC)
				self.a_id = self.graph.DrawImage(data=cv2.imencode('.png', self.disparity_to_show)[1].tobytes(), location=(0, 362))
				# Рисовние линий
				for i,line in enumerate(self.lines_for_view):
					if line[0] is not None and line[1] is not None:
						
						self.graph.DrawLine((line[0][0],line[0][1]), (line[1][0], line[1][1]), color='purple', width=3)
						self.graph.DrawText(self.letter_dict[i], 
							(line[0][0] + (line[1][0] - line[0][0])/2,line[0][1] + (line[1][1] - line[0][1])/2 ), 
							color = 'white')

				self.graph.TKCanvas.tag_lower(self.a_id)
				#--------------------------------------
				# Конец рисовательной части
				#

				# Update image in window
				imageL,ImageR = resize_rectified_pair(rectified_pair)
				self.window.FindElement('image').Update(data=cv2.imencode('.png', imageL)[1].tobytes())

			 


				# Обработчики событий
				# ---------------------

				if not self.ifCamPi:
					# Нажатие на кнопку "Следующая картинка"
					if event == "Next Picture ->":
						self.index = self.index + 1
						if self.index >= len(self.imageToDisp):
							self.index=0
						imageToDisp = self.imageToDisp[self.index] 
						imgL, imgR = calibrate_two_images(imageToDisp, self.ifCamPi)
						rectified_pair = (imgL, imgR)
				
				if event == "3DCloud Settings":
					try:
						self.Window3D.run(imageToDisp, self.parameters, self.ifCamPi)
					except:
						self.window.FindElement("_output_").Update('')
						print("ERROR:Firstly create disparity map!")
						print(traceback.format_exc()) 
				
				if event == "Show 3D-Points":
					os.system("meshlab output.ply")

				
				# Save settings to file
				if event == 'save settings':
					save_csv("db/settings.csv", self.parameters)
				
				# Delete lines from graph image
				if event == "clear lines":
					try:
						self.lines = []
						self.lines_for_view = []
						self.secondWindow.auto_lines = []
						self.disparity, value_disparity = self.deepMap_updater(imageToDisp, values)
						self.graph.TKCanvas.delete('all')
						self.window.FindElement("_output_").Update('')
						print("All lines is deleted.")
					except:
						self.window.FindElement("_output_").Update('')
						print("ERROR:Something wrong with 'clear lines'")

				# Open window with settings
				if event == 'lineFinder Settings':
					#self.window_cubeFinder(disparity)
					try:
						self.disparity, value_disparity = self.deepMap_updater(imageToDisp, values)
						self.secondWindow.run(self.disparity)
					except:
						self.window.FindElement("_output_").Update('')
						print("ERROR:Firstly create disparity map!")
						print(traceback.format_exc()) 


				# События рисования на графе нажатием кнопки мыши
				if event == "graph":
					x,y = values["graph"]
					#print (f"mouse down at ({x},{y})")
					#print(f"disparity = {self.disparity_value[int(y*720/362)][int(x*1280/640)]}")
					if not self.dragging:
						self.start_p = (x,y)
						self.dragging = True
					else:
						self.end_p = (x,y)
						
				elif event.endswith('+UP') and self.end_p!= None:
					self.start_point = self.start_p
					self.end_point = self.end_p
					print(f"grabbed rectangle from {self.start_point} to {self.end_point}")
					#print(f"grabbed rectangle from {[int(start_p[0]*1280/640), int((abs(start_p[1]-362))*720/362)]} to {[int(end_p[0]*1280/640), int((abs(end_p[1]-362))*720/362)]}")
					if self.start_p != None or self.end_p != None or [self.start_p, self.end_p] != None:
						self.lines.append([[int(self.start_p[0]*1280/640), int((abs(self.start_p[1]-362))*720/362)],\
							[int(self.end_p[0]*1280/640), int((abs(self.end_p[1]-362))*720/362)]])
						
						#Bad style
						if self.lines[-1][0][0]>=1280:
							self.lines[-1][0][0] = 1279
						if self.lines[-1][1][0]>=1280:
							self.lines[-1][1][0] = 1279
						if self.lines[-1][0][1]>=720:
							self.lines[-1][0][1] = 719
						if self.lines[-1][1][1]>=720:
							self.lines[-1][1][1] = 719
							
						self.lines_for_view.append([self.start_p, self.end_p])
					self.start_p, self.end_p = None, None
					self.dragging = False

				#print(lines)
				
				if event == 'create map':
					self.disparity, self.disparity_value = self.deepMap_updater(imageToDisp, values)
					self.pointcloud = self.Window3D.forFormula(imageToDisp, self.parameters, self.ifCamPi)
					#self.secondWindow.auto_lines = []
					#lines = []
					self.window.FindElement("_output_").Update('')
					print("Deep map is created.")

				# Нажатие на кнопку "Вычислить", которая должна вернуть 
				# наименования граней и их размеры(Для линий которые мы сами нарисовали).
				if event == 'find distances':
					try:
						self.findAllDistances(self.lines, self.disparity_value)
					except:
						print("'find distances' dont work")
						print(traceback.format_exc())

			except IndexError:
				print(traceback.format_exc())

			return True

		# В зависимости от того, есть ли расбериПи, мы запускаем соответсвующую опцию
		if self.ifCamPi:
			# The PSG "Event Loop"
			for frame in self.camera.capture_continuous(self.capture, format="bgra", use_video_port=True, resize=(self.cam_width,self.cam_height)): 
				loop_flag = main_loop(frame) 
				if not loop_flag:
					break
		else:
			# Двойное изображение
			
			while True:
				imageToDisp = self.imageToDisp[self.index]
				loop_flag = main_loop(imageToDisp) 
				if not loop_flag:
					break

	# Функция, которая выводит информацию о всех
	# найденных и нарисованных линий
	def findAllDistances(self, lines, value_disparity= None):
		if len(lines) == 0 and len(self.secondWindow.auto_lines) == 0:
			self.window.FindElement("_output_").Update('')
			print("No lines that I can find.")
		else:
			self.window.FindElement("_output_").Update('')
			mul_coef = 1
			k=0
			for i in range(0, len(self.secondWindow.auto_lines), 4):
				print(f"Object {int(i/4) + 1}")
				for j in range(4):
					box0,box1 = self.secondWindow.auto_lines[k+j]
					print(f"{self.letter_dict[k]*mul_coef}{j+1} : {round(self.old_determine_line(value_disparity, [box0,box1]), 2)} mm")
				if self.letter_dict[k] == 'Z':
					k=0
					mul_coef = mul_coef + 1
					continue
				k = k+1
				
			for i in range(len(lines)):
				if lines[i] == None:
					del lines[i]
				elif lines[i][0] == None or lines[i][1] == None:
					del lines[i]			
			
			for i,line in enumerate(lines):
				if i==0:
					print("\nLines:")
				line_size = self.determine_line(value_disparity, line)
				print(f"{self.letter_dict[i]} : {round(line_size,2)} mm")
		

	def deepMap_updater(self,imageToDisp, values):
		imgL, imgR = calibrate_two_images(imageToDisp, self.ifCamPi)
		rectified_pair = (imgL, imgR)
		
		self.parameters['SpklWinSze'] = int(values[0])
		self.parameters['SpcklRng'] = int(values[1])
		self.parameters['UnicRatio'] = int(values[2])
		self.parameters['TxtrThrshld'] = int(values[3])
		self.parameters['NumOfDisp'] = int(values[4]/16)*16
		self.parameters['MinDISP'] = int(values[5])
		self.parameters['PreFiltCap'] = int(values[6]/2)*2+1
		self.parameters['PFS'] = int(values[7]/2)*2+1

		return stereo_depth_map(rectified_pair, self.parameters)

	def old_determine_line(self, disparity_map, line, baseline=0.065, focal=1442, rescale=1):
		A = self.boundary_condition(line[0])
		B = self.boundary_condition(line[1])

		disp1 = disparity_map[line[0][1]][line[0][0]]
		disp2 = disparity_map[line[1][1]][line[1][0]]
		depth1 = baseline *focal/ (rescale * disp1)
		depth2 = baseline *focal/ (rescale * disp2)

		line_size = abs(math.sqrt(pow(B[0] - A[0], 2) + pow(B[1] - A[1],2) + pow(depth2 - depth1, 2)))/2.65

		return line_size

	@staticmethod
	def boundary_condition(A):
		if A[1]>=720:
			A[1] = 719
		if A[0] >= 1280:
			A[0]=1279
		return A
		
	
	def determine_line(self, disparity_map, line, baseline=0.065, focal=1442, rescale=1):
		A = self.boundary_condition(line[0])
		B = self.boundary_condition(line[1])

		points = self.pointcloud
		Xa,Ya,Za = points[A[1]][A[0]]
		Xb,Yb,Zb = points[B[1]][B[0]]
		line_size = abs(math.sqrt(pow(Xb - Xa, 2)+pow(Yb - Ya, 2)+pow(Zb - Za, 2)))*23.46

		return line_size
	
	def straight_determine_line(self, line):
		A = line[0]
		B = line[1]
		
		return abs(math.sqrt(pow((B[0] - A[0])/8.0, 2) + pow((B[1] - A[1])/7.6,2))*1/3)


	def close(self):
		self.window.close()
		#self.vs.stop()
