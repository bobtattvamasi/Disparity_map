import cv2, PySimpleGUI as sg
import base64
import glob
import sys, traceback
import subprocess
import math
# import picamera
# from picamera import PiCamera
import numpy as np
import string
import imutils

# Imports from files
from GUI.baseInterface import baseInterface
from measurements_values import defaultValues
from point_cloud import stereo_depth_map as depth_map, create_points_cloud
from determine_object_helper import convert_to_bytes, calibrate_two_images, stereo_depth_map, contours_finder
from GUI.tools_for_intaface import *
from GUI.autoDetectRectWin import autoDetectRectWin
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

		# Создаем словарь параметров для настройки карты глубины,
		# которые ниже можно будет регулировать через ползунки.
		#self.parameters = {}
		# Считываем значения параметров 
		# из файла .csv
		self.parameters = read_csv("db/settings.csv")

		self.secondWindow = autoDetectRectWin("DarkAmber", 'Auto Detecting Rectangles')

		# Переменная для хранения линий
		# с автоопределения прямоугольников
		self.auto_lines = []

		# Словарь параметров для настройки изображений, 
		# которые используются для автоопределения граней
		# прямоугольных  объектов.
		self.secondWin_parameters = read_csv("db/secondWin.csv")

		# Словарь букв для отображения линий
		self.letter_dict = dict(zip([i for i in range(0,26)],string.ascii_uppercase))

		# self.secondWin_parameters = {"lowH":0,
		# 				"highH":179,
		# 				"lowS":99,
		# 				"highS":255,
		# 				"lowV":133,
		# 				"highV":255}


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

		
		
	def lineFinder(self, image):
		hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		self.secondWindow.auto_lines = []
		
		minVal = 50
		maxVal = 200
		layer = 0

		Thmin = self.secondWin_parameters["Thmin"]
		Thmax = self.secondWin_parameters["Thmax"]

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
		j=0
		mul_coef = 1
		cX = 0
		cY = 0
		for i,c in enumerate(cnts):
			# compute the center of the contour, then detect the name of the
			# shape using only the contour
			M = cv2.moments(c)
			if M["m00"] != 0:
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
			
			cv2.putText(image, str(self.letter_dict[j]*mul_coef)+str(1), (int(box[0][0] + (box[1][0] - box[0][0])/2),int(box[0][1] + (box[1][1] - box[0][1])/2) ), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
			fontScale=0.5, color=(255,255,255))
			self.secondWindow.auto_lines.append([box[0],box[1]])
			print(f"{self.letter_dict[j]*mul_coef}1 : {round(self.straight_determine_line([box[0],box[1]]), 2)} mm")
			cv2.putText(image, str(self.letter_dict[j]*mul_coef)+str(2), (int(box[1][0] + (box[2][0] - box[1][0])/2),int(box[1][1] + (box[2][1] - box[1][1])/2) ), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
			fontScale=0.5, color=(255,255,255))
			self.secondWindow.auto_lines.append([box[1],box[2]])
			print(f"{self.letter_dict[j]*mul_coef}2 : {round(self.straight_determine_line([box[1],box[2]]), 2)} mm")
			cv2.putText(image, str(self.letter_dict[j]*mul_coef)+str(3), (int(box[2][0] + (box[3][0] - box[2][0])/2),int(box[2][1] + (box[3][1] - box[2][1])/2) ), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
			fontScale=0.5, color=(255,255,255))
			self.secondWindow.auto_lines.append([box[2],box[3]])
			print(f"{self.letter_dict[j]*mul_coef}3 : {round(self.straight_determine_line([box[2],box[3]]), 2)} mm")
			cv2.putText(image, str(self.letter_dict[j]*mul_coef)+str(4), (int(box[3][0] + (box[0][0] - box[3][0])/2),int(box[3][1] + (box[0][1] - box[3][1])/2) ), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
			fontScale=0.5, color=(255,255,255))
			self.secondWindow.auto_lines.append([box[3],box[0]])
			print(f"{self.letter_dict[j]*mul_coef}4 : {round(self.straight_determine_line([box[3],box[0]]), 2)} mm")
			image = cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
			j=j+1

			if self.letter_dict[j] == 'z':
				j=0
				mul_coef = mul_coef + 1
			
		return image


	# Основная функция в котором работает наше окно
	def run(self):
		# Двойное изображение
		imageToDisp = self.imageToDisp[self.index]

		

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
		

		# The PSG "Event Loop"
		#for frame in self.camera.capture_continuous(self.capture, format="bgra", use_video_port=True, resize=(self.cam_width,self.cam_height)):                     
		while True:
			event, values = self.window.Read(timeout=20, timeout_key='timeout')      # get events for the window with 20ms max wait

			if event is None or event == self.sg.WIN_CLOSED or event == 'Cancel':  
				break  

			#imageToDisp = self.vs.read()
			
			#imageToDisp = frame
			#imgLeft = frame [0:362,0:640]
			#print(f"frame = {imageToDisp.shape}")
			#leftI = frame[0:720, 0:1280]
			#leftI = cv2.resize(leftI, (640,362))
			#rightI = imageToDisp[0:720, 1280:2560]
			#rightI = cv2.resize(rightI, (640,362))

			imgL, imgR = calibrate_two_images(imageToDisp)
			# ~ rectified_pair = (imgR, imgL)
			
			try:

				#disparity, value_disparity = self.deepMap_updater(imageToDisp, values)
				
				if event == 'find lines':

					try:
						disparity, value_disparity = self.deepMap_updater(imageToDisp, values)
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
				save_csv("db/settings.csv", self.parameters)
			
			# Delete lines from graph image
			if event == "clear lines":
				try:
					lines = []
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
				#print (f"mouse down at ({x},{y})")
				if not dragging:
					start_p = (x,y)
					dragging = True
				else:
					end_p = (x,y)
					
			elif event.endswith('+UP') and end_p!= None:
				start_point = start_p
				end_point = end_p
				print(f"grabbed rectangle from {start_point} to {end_point}")
				if start_p != None or end_p != None or [start_p, end_p] != None:
					lines.append([start_p, end_p])
				start_p, end_p = None, None
				dragging = False

			#print(lines)
			
			if event == 'create map':
				disparity, value_disparity = self.deepMap_updater(imageToDisp, values)
				self.secondWindow.auto_lines = []
				lines = []
				self.window.FindElement("_output_").Update('')
				print("Deep map is created.")

			# Нажатие на кнопку "Вычислить", которая должна вернуть 
			# наименования граней и их размеры(Для линий которые мы сами нарисовали).
			if event == 'find length':
				try:
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
								box0,box1 = self.secondWindow.auto_lines[i+j]
								print(f"{self.letter_dict[k]*mul_coef}{j+1} : {round(self.straight_determine_line([box0,box1]), 2)} mm")
							if self.letter_dict[k] == 'z':
								k=0
								mul_coef = mul_coef + 1
								continue

							k = k+1
							

						for i in range(len(lines)):
							if lines[i] == None:
								del lines[i]
							elif lines[i][0] == None or lines[i][1] == None:
								del lines[i]
						#print(f"lines2 = {lines}")
						
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
