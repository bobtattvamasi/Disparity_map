import cv2, PySimpleGUI as sg
import base64
import glob
import sys, traceback, os
import subprocess
import math
#import picamera
#from picamera import PiCamera
import numpy as np
import imutils

# Imports from files
from GUI.baseInterface import baseInterface
from measurements_values import defaultValues
from point_cloud import stereo_depth_map as depth_map, create_points_cloud
from determine_object_helper import convert_to_bytes, calibrate_two_images, stereo_depth_map, contours_finder, autoFindRect, resize_rectified_pair
from GUI.tools_for_intaface import *
from GUI.autoDetectRectWin import autoDetectRectWin
from GUI.threedPointCloudWin import ThreeDPointCloudWin
#from FPS_test import PiVideoStream


  

# Основное окно
class Interface(baseInterface):

	# Функия считывает все картинки из папки
	def image_to_disp(self,path_to_imagesFolder):
		photo_files = []
		i = 0
		for file in glob.glob(path_to_imagesFolder):
			photo_files.append(file)
		return photo_files, i


	def __init__(self, themeStyle, TextForApp):
		super().__init__(themeStyle, TextForApp)

		# Список имен изображений, индекс-указатель на определенное
		# изображение
		self.imageToDisp, self.index = self.image_to_disp(defaultValues.folderTestScenes)
		
		# Camera settimgs
		self.cam_width = 2560
		self.cam_height = 720

		# Считываем значения параметров(Ползунки)
		# из файла .csv
		self.parameters = read_csv("db/settings.csv")

		# Дополнительные окна настроек
		self.secondWindow = autoDetectRectWin("DarkAmber", 'Auto Detecting Rectangles')
		self.Window3D = ThreeDPointCloudWin("DarkAmber", "3D Point Cloud Builder")

		# Переменная для хранения линий
		# с автоопределения прямоугольников
		self.auto_lines = []

		# Словарь параметров для настройки изображений, 
		# которые используются для автоопределения граней
		# прямоугольных  объектов.
		self.secondWin_parameters = read_csv("db/secondWin.csv")

		self.disparity_value = None

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
			# Кнопка котороя показывает следующее изображение
			self.sg.Button('Next Picture ->', size=(15,2)),
			# Кнопка котороя delete lines
			self.sg.Button('clear lines', size=(15,2))
			]
		]

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

		self.pointcloud = None

		
		
	def lineFinder(self, image):
		hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		self.secondWindow.auto_lines = []
		_, mask_frame, thresh = autoFindRect(image, hsv_frame, self, True)	
		return image


	# Основная функция в котором работает наше окно
	def run(self):
		# Двойное изображение
		imageToDisp = self.imageToDisp[self.index]		

		# Калибровка и разделение изображений на левое и правое
		imgL, imgR = calibrate_two_images(imageToDisp, False)
		rectified_pair = (imgL, imgR)

		# Флаг для перерисовывания графа(для возможности рисовать на нем)
		a_id = None
		# Сам граф
		graph = self.window.Element("graph")

		# Флаг и переменные для начала и конца рисования линии
		dragging = False
		start_point, end_point = None, None
		start_p, end_p = None, None
		lines =[]
		lines_for_view = []
		disparity_to_show = np.zeros((362, 640), np.uint8)

		disparity = np.zeros((720,1280), np.uint8)
                
		while True:
			event, values = self.window.Read(timeout=20, timeout_key='timeout')      # get events for the window with 20ms max wait

			if event is None or event == self.sg.WIN_CLOSED or event == 'Cancel':  
				break  

			#imageToDisp = self.vs.read()
			
			#imageToDisp = frame

			imgL, imgR = calibrate_two_images(imageToDisp, False)
			# ~ rectified_pair = (imgR, imgL)
			
			try:

				#disparity, self.disparity_value = self.deepMap_updater(imageToDisp, values)
				
				if event == 'auto-lineFinder':
					self.window.FindElement("_output_").Update('')
					try:
						disparity, value_disparity = self.deepMap_updater(imageToDisp, values)
						self.pointcloud = self.Window3D.forFormula(imageToDisp, self.parameters, ifCamPi=False)
						disparity = self.lineFinder(disparity)
					except:
						print("find lines dont work")
						print(traceback.format_exc())
				
				#
				# Рисовательная часть(возможно не нужно)
				#--------------------------------------
				# Перерисовка графа
				if a_id:
					graph.DeleteFigure(a_id)
				disparity_to_show = cv2.resize(disparity, dsize=(640, 362), interpolation = cv2.INTER_CUBIC)
				a_id = graph.DrawImage(data=cv2.imencode('.png', disparity_to_show)[1].tobytes(), location=(0, 362))
				# Рисовние линий
				print(f"len={len(lines_for_view)}"
				for i,line in enumerate(lines_for_view):
					if line[0] is not None and line[1] is not None:
						
						graph.DrawLine((line[0][0],line[0][1]), (line[1][0], line[1][1]), color='purple', width=3)
						graph.DrawText(self.letter_dict[i], 
							(line[0][0] + (line[1][0] - line[0][0])/2,line[0][1] + (line[1][1] - line[0][1])/2 ), 
							color = 'white')

				graph.TKCanvas.tag_lower(a_id)
				#--------------------------------------
				# Конец рисовательной части
				#

				# Update image in window
				imageL,ImageR = resize_rectified_pair(rectified_pair)
				self.window.FindElement('image').Update(data=cv2.imencode('.png', imageL)[1].tobytes())

			except IndexError:
				print(traceback.format_exc()) 


			# Обработчики событий
			# ---------------------

			if event == "3DCloud Settings":
				try:
					#disparity, points_3, colors = create_points_cloud(imageToDisp, self.parameters)
					self.Window3D.run(imageToDisp, self.parameters, ifCamPi=False)
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
					lines = []
					lines_for_view = []
					self.secondWindow.auto_lines = []
					disparity, value_disparity = self.deepMap_updater(imageToDisp, values)
					graph.TKCanvas.delete('all')
					self.window.FindElement("_output_").Update('')
					print("All lines is deleted.")
				except:
					self.window.FindElement("_output_").Update('')
					print("ERROR:Something wrong with 'clear lines'")

			# Open window with settings
			if event == 'lineFinder Settings':
				#self.window_cubeFinder(disparity)
				try:
					disparity, value_disparity = self.deepMap_updater(imageToDisp, values)
					self.secondWindow.run(disparity)
				except:
					self.window.FindElement("_output_").Update('')
					print("ERROR:Firstly create disparity map!")
					print(traceback.format_exc()) 
			
			

			# Нажатие на кнопку "Следующая картинка"
			if event == "Next Picture ->":
				self.index = self.index + 1
				if self.index >= len(self.imageToDisp):
					self.index=0
				imageToDisp = self.imageToDisp[self.index] 
				imgL, imgR = calibrate_two_images(imageToDisp, False)
				rectified_pair = (imgL, imgR)
				#pass

			# События рисования на графе нажатием кнопки мыши
			if event == "graph":
				x,y = values["graph"]
				#print (f"mouse down at ({x},{y})")
				print(f"disparity = {self.disparity_value[int(y*720/362)][int(x*1280/640)]}")
				#x,y = int(x*1280/640), int(y*720/362)
				if not dragging:
					start_p = (x,y)
					dragging = True
				else:
					end_p = (x,y)
					
			elif event.endswith('+UP') and end_p!= None:
				start_point = start_p
				end_point = end_p
				print(f"grabbed rectangle from {start_point} to {end_point}")
				print(f"grabbed rectangle from {[int(start_p[0]*1280/640), int((abs(start_p[1]-362))*720/362)]} to {[int(end_p[0]*1280/640), int((abs(end_p[1]-362))*720/362)]}")
				if start_p != None or end_p != None or [start_p, end_p] != None:
					lines.append([[int(start_p[0]*1280/640), int((abs(start_p[1]-362))*720/362)], [int(end_p[0]*1280/640), int((abs(end_p[1]-362))*720/362)]])
					lines_for_view.append([start_p, end_p])
				start_p, end_p = None, None
				dragging = False

			#print(lines)
			
			if event == 'create map':
				disparity, self.disparity_value = self.deepMap_updater(imageToDisp, values)
				self.pointcloud = self.Window3D.forFormula(imageToDisp, self.parameters, ifCamPi=False)
				#self.secondWindow.auto_lines = []
				#lines = []
				self.window.FindElement("_output_").Update('')
				print("Deep map is created.")

			# Нажатие на кнопку "Вычислить", которая должна вернуть 
			# наименования граней и их размеры(Для линий которые мы сами нарисовали).
			if event == 'find distances':
				try:
					self.findAllDistances(self.determine_line, lines, self.disparity_value)
				except:
					print("'find distances' dont work")
					print(traceback.format_exc())

	def findAllDistances(self, func, lines, value_disparity= None):
		if len(lines) == 0 and len(self.secondWindow.auto_lines) == 0:
			self.window.FindElement("_output_").Update('')
			print("No lines that I can to find.")
		else:
			self.window.FindElement("_output_").Update('')
			mul_coef = 1
			k=0
			for i in range(0, len(self.secondWindow.auto_lines), 4):
				print(f"Object {int(i/4) + 1}")
				for j in range(4):
					box0,box1 = self.secondWindow.auto_lines[k+j]
					print(f"{self.letter_dict[k]*mul_coef}{j+1} : {round(func(value_disparity, [box0,box1]), 2)} mm")
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
				#line_size = self.determine_line(value_disparity, line)
				line_size = func(value_disparity, line)
				print(f"\n{self.letter_dict[i]} : {round(line_size,2)} mm")

	def deepMap_updater(self,imageToDisp, values):
		imgL, imgR = calibrate_two_images(imageToDisp, ifCamPi=False)
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

	# def determine_line(self, disparity_map, line, baseline=0.065, focal=1442, rescale=1):
	# 	A = line[0]
	# 	B = line[1]

	# 	disp1 = disparity_map[line[0][1]][line[0][0]]
	# 	disp2 = disparity_map[line[1][1]][line[1][0]]
	# 	depth1 = baseline *focal/ (rescale * disp1)
	# 	depth2 = baseline *focal/ (rescale * disp2)
	# 	print(f"disp1 = {disp1} \n disp2 = {disp2}")

	# 	line_size = abs(math.sqrt(pow(B[0] - A[0], 2) + pow(B[1] - A[1],2) + pow(depth2 - depth1, 2)))/2.65

	# 	return line_size

	def determine_line(self, disparity_map, line, baseline=0.065, focal=1442, rescale=1):
		A = line[0]
		B = line[1]

		points = self.pointcloud
		Xa,Ya,Za = points[A[1]][A[0]]
		Xb,Yb,Zb = points[B[1]][B[0]]
		line_size = abs(math.sqrt(pow(Xb - Xa, 2)+pow(Yb - Ya, 2)+pow(Zb - Za, 2)))

		return line_size
	
	def straight_determine_line(self, line):
		A = line[0]
		B = line[1]
		
		return abs(math.sqrt(pow((B[0] - A[0])/8.0, 2) + pow((B[1] - A[1])/7.6,2)))


	def close(self):
		self.window.close()
		#self.vs.stop()
