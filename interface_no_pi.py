import cv2, PySimpleGUI as sg
from determine_object_helper import convert_to_bytes, calibrate_two_images, stereo_depth_map, contours_finder
import base64
import glob
import sys, traceback
from point_cloud import stereo_depth_map as depth_map, create_points_cloud
from abc import ABC, abstractmethod
import subprocess
import math
# import picamera
# from picamera import PiCamera
import numpy as np
from measurements_values import defaultValues
import string
import imutils

#from FPS_test import PiVideoStream
import csv


# Базовый класс интерфейса
class baseInterface(ABC):
	def __init__(self, themeStyle, TextForApp):
		self.sg = sg
		self.sg.theme(themeStyle)
		self.icon = "data/8.ico"
		layout = []
		self.window = self.sg.Window(TextForApp, 
						layout, icon=self.icon, resizable=True)


	@abstractmethod
	def run(self):
		while True:                     # The PSG "Event Loop"
			event, values = self.window.Read(timeout=20, timeout_key='timeout')      # get events for the window with 20ms max wait

			if event is None or event == self.sg.WIN_CLOSED or event == 'Cancel':  
				break     

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
		# Final image capture settings
		# scale_ratio = 1
		# # Camera resolution height must be dividable by 16, and width by 32
		# cam_width = int((self.cam_width+31)/32)*32
		# cam_height = int((self.cam_height+15)/16)*16
		# print ("Camera resolution: "+str(cam_width)+" x "+str(cam_height))

		# # Buffer for captured image settings
		# self.img_width = int (cam_width * scale_ratio)
		# self.img_height = int (cam_height * scale_ratio)
		# self.capture = np.zeros((self.img_height, self.img_width, 4), dtype=np.uint8)
		# print ("Scaled image resolution: "+str(self.img_width)+" x "+str(self.img_height))

		# # Initialize the camera
		# self.camera = PiCamera(stereo_mode='side-by-side',stereo_decimate=False)
		# self.camera.resolution=(cam_width, cam_height)
		# self.camera.framerate = 20
		# self.camera.hflip = True
		#self.vs = PiVideoStream(resolution=(self.cam_width,self.cam_height)).start()
		self.parameters = {}
		#try:
		reader = csv.reader(open("settings.csv", 'r'))
		self.auto_lines = []
		file_flag = False
		for row in reader:
			k,v = row
			self.parameters[k] = int(v)
		#if file_flag:
		# ~ self.parameters = {'SpklWinSze':68,
			# ~ 'SpcklRng': 8,
			# ~ 'UnicRatio':8,
			# ~ 'TxtrThrshld':0,
			# ~ 'NumOfDisp':53,
			# ~ 'MinDISP':-126,
			# ~ 'PreFiltCap':16,
			# ~ 'PFS': 5
			# ~ #'SWS': 5
			# ~ }


		# --------------------------------------------
		# Разделяем пространство окна на левое и правое
		#
		# Левая колонка:
		left_column = [
			# Место под левое изображение
			[self.sg.Image(filename='', key='image')],
			# Далее идут слайдеры параметров для настройки выходной карты глубин
			[self.sg.Frame('Settings',[
			[self.sg.Text("SpklWinSze"), self.sg.Slider(range=(0, 300), orientation='h', size=(34, 10), default_value=self.parameters["SpklWinSze"])],
			[self.sg.Text("SpcklRng"), self.sg.Slider(range=(0, 40), orientation='h', size=(34, 10), default_value=self.parameters["SpcklRng"])],
			[self.sg.Text("UnicRatio"), self.sg.Slider(range=(1, 80), orientation='h', size=(34, 10), default_value=self.parameters["UnicRatio"])],
			[self.sg.Text("TxtrThrshld"), self.sg.Slider(range=(0, 1000), orientation='h', size=(34, 10), default_value=self.parameters["TxtrThrshld"])],
			[self.sg.Text("NumOfDisp"), self.sg.Slider(range=(16, 256), orientation='h', size=(34, 10), default_value=self.parameters["NumOfDisp"])],
			[self.sg.Text("MinDISP"), self.sg.Slider(range=(-300, 300), orientation='h', size=(34, 10), default_value=self.parameters["MinDISP"])],
			[self.sg.Text("PreFiltCap"), self.sg.Slider(range=(5, 63), orientation='h', size=(34, 10), default_value=self.parameters["PreFiltCap"])],
			[self.sg.Text("PFS"), self.sg.Slider(range=(5, 255), orientation='h', size=(34, 10), default_value=self.parameters["PFS"])],
			#[self.sg.Text("SWS"), self.sg.Slider(range=(5, 255), orientation='h', size=(34, 10), default_value=5)]
			[self.sg.Button("save settings", size=(10,1))]
			])],
			# button for create map of disparity
			[self.sg.Button('create map', size=(15,2)),
			# Кнопка по которой вычисляются размеры найденных граней
			self.sg.Button('find length', size=(15,2)),
			# Кнопка котороя показывает следующее изображение
			#self.sg.Button('Next Picture ->', size=(15,2))
			# Кнопка котороя delete lines
			self.sg.Button('clear lines', size=(15,2))
			]
		]

		# Правая колонка:
		#
		right_column = [
			# Место для отображения второй картинки
			# Только в виде графа, что бы можно было 
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
			[self.sg.Output(size=(64, 12), key = '_output_')],
			[self.sg.Button('lineFinder Settings', size=(15,2)), self.sg.Button('find lines', size=(15,2))]
			
		]

		# Объединяемвсе в один объект
		layout  = [[self.sg.Column(left_column, element_justification='c'), 
					self.sg.VSeperator(),
					self.sg.Column(right_column, element_justification='c')]
				]
		# Наше окно
		self.window = self.sg.Window(TextForApp, 
						layout, icon=self.icon, resizable=True)

		# Словарь букв для отображения линий
		self.letter_dict = dict(zip([i for i in range(0,26)],string.ascii_uppercase))
		
		self.secondWin_parameters = {"lowH":0,
						"highH":179,
						"lowS":99,
						"highS":255,
						"lowV":133,
						"highV":255}
		
	def lineFinder(self, image):
		hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		self.auto_lines = []
		
		minVal = 50
		maxVal = 200
		layer = 0

		Thmin = 60
		Thmax = 255

		hsv_min = np.array((self.secondWin_parameters["lowH"], self.secondWin_parameters["lowS"], self.secondWin_parameters["lowV"]), np.uint8)
		hsv_max = np.array((self.secondWin_parameters["highH"], self.secondWin_parameters["highS"], self.secondWin_parameters["highV"]), np.uint8)
		
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		height, width = gray.shape
		mask = np.zeros((height,width))
		cv2.rectangle(mask, (80,140), (460,355), 255, -1)
		#masked_data = cv2.bitwise_and(gray, gray, mask=mask)
		gray[mask < 255] = 0
		hsv_frame[mask < 255] = 0
		mask_frame = cv2.inRange(hsv_frame, hsv_min, hsv_max)
		#img = image[140:355, 80:460]
		
		#gray = image
		blurred = cv2.GaussianBlur(mask_frame, (5, 5), 0)
		thresh = cv2.threshold(blurred, Thmin, Thmax, cv2.THRESH_BINARY)[1]

		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		ratio =1
		boxes = []
		#img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
		for i,c in enumerate(cnts):
			# compute the center of the contour, then detect the name of the
			# shape using only the contour
			M = cv2.moments(c)
			cX = int((M["m10"] / M["m00"]) * ratio)
			cY = int((M["m01"] / M["m00"]) * ratio)
			# multiply the contour (x, y)-coordinates by the resize ratio,
			# then draw the contours and the name of the shape on the image
			c = c.astype("float")
			c *= ratio
			c = c.astype("int")
			contour_area = cv2.contourArea(c)
			if contour_area < 190 or contour_area > 2400:
				continue
			#print(f"{i} : area = {contour_area}")
			peri = cv2.arcLength(c, True)
			c = cv2.approxPolyDP(c, 0.020 * peri, True)
			# rotated rect constructor
			rect = cv2.minAreaRect(c)
			
			box = np.int0(cv2.boxPoints(rect))
			#print(f"recr = {box}")
			#-------------------------
			
			cv2.putText(image, str(self.letter_dict[i])+str(1), (int(box[0][0] + (box[1][0] - box[0][0])/2),int(box[0][1] + (box[1][1] - box[0][1])/2) ), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
			fontScale=0.5, color=(255,255,255))
			self.auto_lines.append([box[0],box[1]])
			print(f"{self.letter_dict[i]}1 : {self.straight_determine_line([box[0],box[1]])}")
			cv2.putText(image, str(self.letter_dict[i])+str(2), (int(box[1][0] + (box[2][0] - box[1][0])/2),int(box[1][1] + (box[2][1] - box[1][1])/2) ), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
			fontScale=0.5, color=(255,255,255))
			self.auto_lines.append([box[1],box[2]])
			print(f"{self.letter_dict[i]}2 : {self.straight_determine_line([box[1],box[2]])}")
			cv2.putText(image, str(self.letter_dict[i])+str(3), (int(box[2][0] + (box[3][0] - box[2][0])/2),int(box[2][1] + (box[3][1] - box[2][1])/2) ), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
			fontScale=0.5, color=(255,255,255))
			self.auto_lines.append([box[2],box[3]])
			print(f"{self.letter_dict[i]}3 : {self.straight_determine_line([box[2],box[3]])}")
			cv2.putText(image, str(self.letter_dict[i])+str(4), (int(box[3][0] + (box[0][0] - box[3][0])/2),int(box[3][1] + (box[0][1] - box[3][1])/2) ), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
			fontScale=0.5, color=(255,255,255))
			self.auto_lines.append([box[3],box[0]])
			print(f"{self.letter_dict[i]}4 : {self.straight_determine_line([box[3],box[0]])}")
			image = cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
			
		return image
		

	def window_cubeFinder(self, image):


		left_column = [[sg.Image(filename='', key='cube_image'), sg.Image(filename='', key='another_image')],
					[sg.Text("lowH"), sg.Slider(range=(0, 179), orientation='h', size=(34, 10), default_value=self.secondWin_parameters["lowH"])],
					[sg.Text("highH"), sg.Slider(range=(0, 179), orientation='h', size=(34, 10), default_value=self.secondWin_parameters["highH"])],
					[sg.Text("lowS"), sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=self.secondWin_parameters["lowS"])],
					[sg.Text("highS"), sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=self.secondWin_parameters["highS"])],
					[sg.Text("lowV"), sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=self.secondWin_parameters["lowV"])],
					[sg.Text("highV"), sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=self.secondWin_parameters["highV"])],
			[sg.Text("minVal"), sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=50)],
			[sg.Text("maxVal"), sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=200)],
			[sg.Text("Layer"), sg.Slider(range=(0, 7), orientation='h', size=(34, 10), default_value=0)],
				[sg.Text("ThMin"), sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=60)],
				[sg.Text("ThMax"), sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=255)],
				[sg.Button('find contours', size=(15,2))]
					]
		right_column = [[sg.Image(filename='', key='mask_contor_image')],
						[sg.Image(filename='', key='hsv_image')]
						]
		layout  = [[sg.Column(left_column, element_justification='c'), 
					sg.VSeperator(),
					sg.Column(right_column, element_justification='c')]
					]
		window1 = sg.Window("Win1", layout)
		hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		self.auto_lines = []
		while True:
			event, values = window1.read(timeout=200)
			if not event:
				break

			self.secondWin_parameters["lowH"] = int(values[0])
			self.secondWin_parameters["highH"] = int(values[1])
			self.secondWin_parameters["lowS"] = int(values[2])
			self.secondWin_parameters["highS"] = int(values[3])
			self.secondWin_parameters["lowV"] = int(values[4])
			self.secondWin_parameters["highV"] = int(values[5])

			minVal = int(values[6])
			maxVal = int(values[7])
			layer = int(values[8])

			Thmin = int(values[9])
			Thmax = int(values[10])

			hsv_min = np.array((self.secondWin_parameters["lowH"], self.secondWin_parameters["lowS"], self.secondWin_parameters["lowV"]), np.uint8)
			hsv_max = np.array((self.secondWin_parameters["highH"], self.secondWin_parameters["highS"], self.secondWin_parameters["highV"]), np.uint8)

			

			#mask_frame = contours_finder(mask_frame, minVal, maxVal, layer, -1)

			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			height, width = gray.shape
			mask = np.zeros((height,width))
			cv2.rectangle(mask, (80,140), (460,355), 255, -1)
			#masked_data = cv2.bitwise_and(gray, gray, mask=mask)
			gray[mask < 255] = 0
			hsv_frame[mask < 255] = 0
			mask_frame = cv2.inRange(hsv_frame, hsv_min, hsv_max)
			#img = image[140:355, 80:460]
			
			#gray = image
			blurred = cv2.GaussianBlur(mask_frame, (5, 5), 0)
			thresh = cv2.threshold(blurred, Thmin, Thmax, cv2.THRESH_BINARY)[1]

			cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
			cnts = imutils.grab_contours(cnts)
			ratio =1
			boxes = []
			#img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
			for i,c in enumerate(cnts):
				# compute the center of the contour, then detect the name of the
				# shape using only the contour
				M = cv2.moments(c)
				cX = int((M["m10"] / M["m00"]) * ratio)
				cY = int((M["m01"] / M["m00"]) * ratio)
				# multiply the contour (x, y)-coordinates by the resize ratio,
				# then draw the contours and the name of the shape on the image
				c = c.astype("float")
				c *= ratio
				c = c.astype("int")
				contour_area = cv2.contourArea(c)
				if contour_area < 190 or contour_area > 2400:
					continue
				#print(f"{i} : area = {contour_area}")
				peri = cv2.arcLength(c, True)
				c = cv2.approxPolyDP(c, 0.020 * peri, True)
				# rotated rect constructor
				rect = cv2.minAreaRect(c)
				
				box = np.int0(cv2.boxPoints(rect))
				#print(f"recr = {box}")
				#-------------------------
				
				self.auto_lines.append([box[0],box[1]])
				self.auto_lines.append([box[1],box[2]])
				self.auto_lines.append([box[2],box[3]])
				self.auto_lines.append([box[3],box[0]])
				
				# rect constructor ------
				(x,y,w,h) = cv2.boundingRect(c)
				boxes.append([x,y,x+w, y+h])
				cv2.rectangle(image, (x,y), (x+w,y+h),(0,0,255),2)
				# -----------------------
				image = cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
			
				

			

			window1.FindElement('cube_image').Update(data=cv2.imencode('.png', image)[1].tobytes())
			window1.FindElement('mask_contor_image').Update(data=cv2.imencode('.png', gray)[1].tobytes())
			window1.FindElement('hsv_image').Update(data=cv2.imencode('.png', mask_frame)[1].tobytes())


	# Основная функция в котором работает наше окно
	def run(self):
		# Двойное изображение
		imageToDisp = self.imageToDisp[self.index]

		# Создаем словарь параметров для настройки карты глубины,
		# коорые ниже можно будет регулировать через ползунки.
		# ~ parameters = {'SpklWinSze':0,
				# ~ 'SpcklRng': 0,
				# ~ 'UnicRatio':25,
				# ~ 'TxtrThrshld':0,
				# ~ 'NumOfDisp':80,
				# ~ 'MinDISP':61,
				# ~ 'PreFiltCap':5,
				# ~ 'PFS': 5
				# ~ #'SWS': 5
				# ~ }

		# Калибровка и разделение изображений на левое и правое
		imgL, imgR = calibrate_two_images(imageToDisp)
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
		disparity = np.zeros((362, 640), np.uint8)
		#cv2.imwrite("mask.jpg", disparity)
		

		# The PSG "Event Loop"
		#for frame in self.camera.capture_continuous(self.capture, format="bgra", use_video_port=True, resize=(self.cam_width,self.cam_height)):                     
		while True:
			event, values = self.window.Read(timeout=20, timeout_key='timeout')      # get events for the window with 20ms max wait

			if event is None or event == self.sg.WIN_CLOSED or event == 'Cancel':  
				break  
			#imageToDisp = self.vs.read()
			#cv2.imwrite("thread_pic.jpg", imageToDisp)
			#frame =cv2.imread(imageToDisp)
			#imageToDisp = frame
			#imgLeft = frame [0:362,0:640]
			#print(f"frame = {imageToDisp.shape}")
			# ~ leftI = frame[0:720, 0:1280]
			# ~ leftI = cv2.resize(leftI, (640,362))
			
			# ~ rightI = imageToDisp[0:720, 1280:2560]
			# ~ rightI = cv2.resize(rightI, (640,362))
			imgL, imgR = calibrate_two_images(imageToDisp)
			# ~ rectified_pair = (imgR, imgL)
			#print(f"shape = {imgR.shape}")
			
			try:
				# ~ parameters['SpklWinSze'] = int(values[0])
				# ~ parameters['SpcklRng'] = int(values[1])
				# ~ parameters['UnicRatio'] = int(values[2])
				# ~ parameters['TxtrThrshld'] = int(values[3])
				# ~ parameters['NumOfDisp'] = int(values[4]/16)*16
				# ~ parameters['MinDISP'] = int(values[5])
				# ~ parameters['PreFiltCap'] = int(values[6]/2)*2+1
				# ~ parameters['PFS'] = int(values[7]/2)*2+1
				# ~ #parameters['SWS'] = int(values[8]/2)*2+1

				# ~ minVal = int(values[9])
				# ~ maxVal = int(values[10])
				# ~ layer  = int(values[11])

				# ~ # Получение карты глубины
				# ~ #disparity, value_disparity = stereo_depth_map(rectified_pair, parameters)

				# ~ # Нахождение на карте глубин границ и отображение их
				# ~ #disparity = contours_finder(disparity, minVal, maxVal, layer, -1)

				# ~ imgL, imgR = calibrate_two_images(imageToDisp)
				# ~ rectified_pair = ( imgL, imgR)
				
				# self.parameters['SpklWinSze'] = int(values[0])
				# self.parameters['SpcklRng'] = int(values[1])
				# self.parameters['UnicRatio'] = int(values[2])
				# self.parameters['TxtrThrshld'] = int(values[3])
				# self.parameters['NumOfDisp'] = int(values[4]/16)*16
				# self.parameters['MinDISP'] = int(values[5])
				# self.parameters['PreFiltCap'] = int(values[6]/2)*2+1
				# self.parameters['PFS'] = int(values[7]/2)*2+1
				#parameters['SWS'] = int(values[8]/2)*2+1

				# minVal = int(values[9])
				# maxVal = int(values[10])
				# layer  = int(values[11])

				#disparity, value_disparity = stereo_depth_map(rectified_pair, self.parameters)
				disparity, value_disparity = self.deepMap_updater(imageToDisp, values)
				
				if event == 'find lines':
					try:
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
				a_id = graph.DrawImage(data=cv2.imencode('.png', disparity)[1].tobytes(), location=(0, 400))
				# Рисовние линий
				for i,line in enumerate(lines):
					if line[0] is not None and line[1] is not None:
						
						graph.DrawLine((line[0][0],line[0][1]), (line[1][0], line[1][1]), color='purple', width=3)
						graph.DrawText(self.letter_dict[i], 
							(line[0][0] + (line[1][0] - line[0][0])/2,line[0][1] + (line[1][1] - line[0][1])/2 ), 
							color = 'white')
				# ~ for i,line in enumerate(self.auto_lines):
					# ~ print(f"line = {line[0][0]}")
					# ~ graph.DrawLine((line[0][0],abs(line[0][1]-362)), (line[1][0], abs(line[1][1]-362)), color='purple', width=3)
					# ~ graph.DrawText(self.letter_dict[i], 
						# ~ (line[0][0] + (abs(line[1][0]-362) - line[0][0])/2,abs(line[0][1]-362) + (abs(line[1][1]-362) - abs(line[0][1]-362))/2 ), 
						# ~ color = 'white')
				#if start_point is not None and end_point is not None:
				#	graph.DrawLine(start_point, end_point, color='purple', width=10)
				graph.TKCanvas.tag_lower(a_id)
				#--------------------------------------
				# Конец рисовательной части
				#

				# Update image in window
				self.window.FindElement('image').Update(data=cv2.imencode('.png', imgL)[1].tobytes())

			except IndexError:
				print(traceback.format_exc()) 


			# Обработчики событий
			# ---------------------


			
			# Save settings to file
			if event == 'save settings':
				with open("settings.csv","w") as f:
					w = csv.writer(f)
					for key, val in self.parameters.items():
						w.writerow([key, val])
				print("settings saved in settings.csv")
			
			# Delete lines from graph image
			if event == "clear lines":
				lines = []
				graph.TKCanvas.delete('all')

			# Open window with settings
			if event == 'lineFinder Settings':
				self.window_cubeFinder(disparity)
			
			

			# Нажатие на кнопку "Следующая картинка"
			# ~ if event == "Next Picture ->":
				# ~ self.index = self.index + 1
				# ~ if self.index >= len(self.imageToDisp):
					# ~ self.index=0
				# ~ imageToDisp = self.imageToDisp[self.index] 
				# ~ imgL, imgR = calibrate_two_images(imageToDisp)
				# ~ rectified_pair = (imgL, imgR)
				# ~ #pass

			# События рисования на графе нажатием кнопки мыши
			if event == "graph":
				x,y = values["graph"]
				print (f"mouse down at ({x},{y})")
				if not dragging:
					start_p = (x,y)
					dragging = True
				else:
					end_p = (x,y)
					
			elif event.endswith('+UP'):
				start_point = start_p
				end_point = end_p
				print(f"grabbed rectangle from {start_point} to {end_point}")
				if start_p != None or end_p != None or [start_p, end_p] != None:
					lines.append([start_p, end_p])
				start_p, end_p = None, None
				dragging = False

			#print(lines)
			
			if event == 'create map':
				# imgL, imgR = calibrate_two_images(imageToDisp)
				# rectified_pair = (imgL, imgR)
				
				# self.parameters['SpklWinSze'] = int(values[0])
				# self.parameters['SpcklRng'] = int(values[1])
				# self.parameters['UnicRatio'] = int(values[2])
				# self.parameters['TxtrThrshld'] = int(values[3])
				# self.parameters['NumOfDisp'] = int(values[4]/16)*16
				# self.parameters['MinDISP'] = int(values[5])
				# self.parameters['PreFiltCap'] = int(values[6]/2)*2+1
				# self.parameters['PFS'] = int(values[7]/2)*2+1

				# disparity, value_disparity = stereo_depth_map(rectified_pair, self.parameters)
				disparity, value_disparity = self.deepMap_updater(imageToDisp, values)

			# Нажатие на кнопку "Вычислить", которая должна вернуть 
			# наименования граней и их размеры(Для линий которые мы сами нарисовали).
			if event == 'find length':
				try:
					for i in range(len(lines)):
						if lines[i] == None:
							del lines[i]
						elif lines[i][0] == None or lines[i][1] == None:
							del lines[i]
					#print(f"lines2 = {lines}")
					
					self.window.FindElement("_output_").Update('')
					
					for i,line in enumerate(lines):
						#line_size = self.determine_line(value_disparity, line)
						line_size = self.straight_determine_line(line)
						print(f"{self.letter_dict[i]} : {round(line_size,2)} mm")
				except:
					print("find length dont work")
					print(traceback.format_exc())

	def deepMap_updater(self,imageToDisp, values):
		imgL, imgR = calibrate_two_images(imageToDisp)
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

	def determine_line(self, disparity_map, line, baseline=0.065, focal=1442, rescale=1):
		A = line[0]
		B = line[1]

		disp1 = disparity_map[line[0][1]][line[0][0]]
		disp2 = disparity_map[line[1][1]][line[1][0]]
		depth1 = baseline *focal/ (rescale * disp1)
		depth2 = baseline *focal/ (rescale * disp2)

		line_size = abs(math.sqrt(pow(B[0] - A[0], 2) + pow(B[1] - A[1],2) + pow(depth2 - depth1, 2)))

		return line_size
	
	def straight_determine_line(self, line):
		A = line[0]
		B = line[1]
		
		return abs(math.sqrt(pow((B[0] - A[0])/8.0, 2) + pow((B[1] - A[1])/7.6,2)))


	def close(self):
		self.window.close()
		#self.vs.stop()
