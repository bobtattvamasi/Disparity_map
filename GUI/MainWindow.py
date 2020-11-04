import cv2, PySimpleGUI as sg
import base64
import glob
import sys, traceback, os
import subprocess
import math

import numpy as np
import imutils

# Imports from files
from GUI.BaseWindow import BaseWindow
from config.config import configValues as cfv
from tools.ImageProccessHelper import calibrate_two_images, stereo_depth_map, resize_rectified_pair, autoFindRect
from GUI.AutoDetectRectWindow import AutoDetectRectWindow
from GUI.PointCloudWindow import PointCloudWindow
#from FPS_test import PiVideoStream

  

# Основное окно
class MainWindow(BaseWindow):

	# Функия считывает все картинки из папки
	@staticmethod
	def __image_to_disp(path_to_imagesFolder):
		photo_files = []
		i = 0
		for file in glob.glob(path_to_imagesFolder):
			photo_files.append(file)
		return photo_files, i

	def __init__(self, themeStyle, TextForApp, ifCamPi=True):
		super().__init__(themeStyle, TextForApp)
		# Параметр отвечающий за то, используется алгоритм со стереокамерой 
		# или с картинками.
		self.ifCamPi = ifCamPi

		# Если со стереокамерой, то загружаем нужную библиотеку
		# и заводим необходимые переменные. 
		if ifCamPi:
			import picamera
			from picamera import PiCamera

			# Camera settimgs
			self.cam_width = cfv.PHOTO_WIDTH
			self.cam_height = cfv.PHOTO_HEIGHT

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
			self.imageToDisp, self.index = self.__image_to_disp(cfv.folderTestScenes)

		# Создаем дочерние окна
		self.autoFinderWindow = AutoDetectRectWindow("DarkAmber", 'Auto Detecting Rectangles')
		self.Window3D = PointCloudWindow("DarkAmber", "3D Point Cloud Builder")

		# Матрица карты несоответсвий
		self.disparity_value = None

		# Для выбора языка
		#
		# левая колонка:
		self.settings = ('Settings', 'Настройки')
		self.save_settings = ('save settings', 'сохранить')
		self.FindClear  = ('Find&Clear', 'Найти/Очистить')
		self.find_distances = ('find distances', 'найти расстояния')
		self.clear_lines = ('clear lines', 'очистить линии')
		self.create_map = ('create map','создать карту')
		if not self.ifCamPi:
			self.Next_Picture = ("Next Picture ->", 'следующее изображение')

		# Правая:
		self.lineFind3D_Settings  = ('lineFind&3D Settings', 'Найтройки автонахождения и 3D')
		self.lineFinder_Settings = ('lineFinder Settings', 'найтройки автонахождения линий')
		self.D3Cloud_Settings = ('3DCloud Settings','настройка 3D')
		self.auto_lineFinder = ('auto-lineFinder', 'Автонахождение объектов')
		
		# --------------------------------------------
		# Разделяем пространство окна на левое и правое
		#
		# Левая колонка:
		left_column = [
			# Место под левое изображение
			[self.sg.Image(filename='', key='image')],
			# Далее идут слайдеры параметров для настройки выходной карты глубин
			[self.sg.Frame(self.settings[self.language],[
			[self.sg.Text("SpklWinSze "), self.sg.Slider(range=(0, 300), orientation='h', size=(34, 10), default_value=self.db.mWinParameters["SpklWinSze"])],
			[self.sg.Text("SpcklRng   "), self.sg.Slider(range=(0, 40), orientation='h', size=(34, 10), default_value=self.db.mWinParameters["SpcklRng"])],
			[self.sg.Text("UnicRatio  "), self.sg.Slider(range=(1, 80), orientation='h', size=(34, 10), default_value=self.db.mWinParameters["UnicRatio"])],
			[self.sg.Text("TxtrThrshld"), self.sg.Slider(range=(0, 1000), orientation='h', size=(34, 10), default_value=self.db.mWinParameters["TxtrThrshld"])],
			[self.sg.Text("NumOfDisp  "), self.sg.Slider(range=(16, 256), orientation='h', size=(34, 10), default_value=self.db.mWinParameters["NumOfDisp"])],
			[self.sg.Text("MinDISP    "), self.sg.Slider(range=(-300, 300), orientation='h', size=(34, 10), default_value=self.db.mWinParameters["MinDISP"])],
			[self.sg.Text("PreFiltCap "), self.sg.Slider(range=(0, 63), orientation='h', size=(34, 10), default_value=self.db.mWinParameters["PreFiltCap"])],
			[self.sg.Text("PFS        "), self.sg.Slider(range=(5, 255), orientation='h', size=(34, 10), default_value=self.db.mWinParameters["PFS"])],
			# ~ [self.sg.Text('_'*25+'DepthFilter'+'_'*20)],
			# ~ [self.sg.Text("lambda"), self.sg.Slider(range=(0, 10000), orientation='h', size=(34, 10), default_value=self.db.mWinParameters['lambda'])],
			# ~ [self.sg.Text("SigmaColor"), self.sg.Slider(range=(0.0, 3.0), orientation='h', size=(34, 10), default_value=self.db.mWinParameters['sigmaColor'], resolution=.1)],
			# ~ [self.sg.Text("Radius"), self.sg.Slider(range=(0, 100), orientation='h', size=(34, 10), default_value=self.db.mWinParameters['Radius'])],
			# ~ [self.sg.Text("LRCthresh"), self.sg.Slider(range=(0, 300), orientation='h', size=(34, 10), default_value=self.db.mWinParameters['LRCthresh'])],
			# sg.Combo(['eng', 'rus'], enable_events=True, key='combo'),
			[self.sg.Button(self.save_settings[self.language], size=(10,1))]
			])],
			
			[self.sg.Frame(self.FindClear[self.language],[
			# Кнопка по которой вычисляются размеры найденных граней
			[self.sg.Button(self.find_distances[self.language], size=(15,2)),
			# Кнопка котороя delete lines
			self.sg.Button(self.clear_lines[self.language], size=(15,2))]
			])],
			# button for create map of disparity
			[self.sg.Button(self.create_map[self.language], size=(15,2))]
		]

		if not self.ifCamPi:
			left_column.append([self.sg.Button(self.Next_Picture[self.language], size=(15,2))])

		# Правая колонка:
		#
		right_column = [
			# Место для отображения второй картинки
			# Только в виде графа, чтобы можно было 
			# рисовать поверх грани, которые мы будем
			# измерять.
			[self.sg.Graph(canvas_size=(cfv.WINDOW_WIDTH, cfv.WINDOW_HEIGHT),
					graph_bottom_left=(0, 0),
					graph_top_right=(cfv.WINDOW_WIDTH, cfv.WINDOW_HEIGHT),
					key="graph",
					change_submits=True, # mouse click events
					drag_submits=True # mouse drag events]
					)],
			# Вывод всей важной информации происходит здесь
			[self.sg.Output(size=(64, 21), key = '_output_')],
			# Группируем кнопки
			[self.sg.Frame(self.lineFind3D_Settings[self.language],[
			# Кнопка для настройки автонахождения линий
			[self.sg.Button(self.lineFinder_Settings[self.language], size=(15,2)), self.sg.Button(self.D3Cloud_Settings[self.language], size=(15,2))]
			])],
			# Кнопка для автонахождения линий на картинке
			[self.sg.Button(self.auto_lineFinder[self.language], size=(15,2))]
			
		]

		# Объединяем колонки в один слой
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
		self.disparity_to_show = np.zeros((cfv.WINDOW_HEIGHT, cfv.WINDOW_WIDTH), np.uint8)
		self.disparity = np.zeros((cfv.IMAGE_HEIGHT,cfv.IMAGE_WIDTH), np.uint8)
		
		
	def lineFinder(self) -> None:
		hsv_frame = cv2.cvtColor(self.disparity, cv2.COLOR_BGR2HSV)
		self.autoFinderWindow.auto_lines = []
		autoFindRect(self.disparity, hsv_frame, self, True)

	# Основная функция в котором работает наше окно
	def run(self, ifDebugVersion=True) -> bool:
		def main_loop(frame):
			event, values = self.window.Read(timeout=20, timeout_key='timeout')      # get events for the window with 20ms max wait

			if event is None or event == self.sg.WIN_CLOSED or event == 'Cancel':  
				return False

			#imageToDisp = self.vs.read()
			
			imageToDisp = frame

			imgL, imgR = calibrate_two_images(imageToDisp, self.ifCamPi)
			rectified_pair = (imgR, imgL)
			
			try:
				# Если мы хотим что бы карта несоответсвий постоянно обновлялась
				if ifDebugVersion:
					self.disparity, self.autoFinderWindow.disparity_value = self.deepMap_updater(imageToDisp, values)
					self.disparity_value = self.autoFinderWindow.disparity_value
				
				# Обновляем граф, рисуя на нем линиями
				self.drawStraightLine()

				# Update image in window
				imageL,ImageR = resize_rectified_pair(rectified_pair)
				self.window.FindElement('image').Update(data=cv2.imencode('.png', imageL)[1].tobytes())


				#----------------------
				# Обработчики событий: |
				# ---------------------

				if not self.ifCamPi:
					# Нажатие на кнопку "Следующая картинка"
					if event == self.Next_Picture[self.language]:
						self.index = self.index + 1
						if self.index >= len(self.imageToDisp):
							self.index=0
						imageToDisp = self.imageToDisp[self.index] 
						imgL, imgR = calibrate_two_images(imageToDisp, self.ifCamPi)
						rectified_pair = (imgL, imgR)

				# Найти и определить контуры объектов
				if event == self.auto_lineFinder[self.language]:
					self.window.FindElement("_output_").Update('')
					try:
						self.disparity, self.autoFinderWindow.disparity_value = self.deepMap_updater(imageToDisp, values)
						self.disparity_value = self.autoFinderWindow.disparity_value
						self.Window3D.updatePointCloud(self.disparity,self.disparity_value, self.ifCamPi)
						self.lineFinder()
					except:
						print("find lines dont work")
						print(traceback.format_exc())
				
				# Открывает окно настроек 3д облака точек
				if event == self.D3Cloud_Settings[self.language]:
					try:
						self.Window3D.run(self.disparity,self.disparity_value, self.ifCamPi)
					except:
						self.window.FindElement("_output_").Update('')
						print("ERROR:Firstly create disparity map!")
						print(traceback.format_exc()) 
				
				# Save settings to file
				if event == self.save_settings[self.language]:
					self.db.save_csv("db/settings.csv", self.db.mWinParameters)
				
				# Delete lines from graph image
				if event == self.clear_lines[self.language]:
					try:
						self.lines = []
						self.lines_for_view = []
						self.autoFinderWindow.auto_lines = []
						self.disparity, value_disparity = self.deepMap_updater(imageToDisp, values)
						self.graph.TKCanvas.delete('all')
						self.window.FindElement("_output_").Update('')
						print("All lines is deleted.")
					except:
						self.window.FindElement("_output_").Update('')
						print("ERROR:Something wrong with 'clear lines'")

				# Open window with lineFinder settings
				if event == self.lineFinder_Settings[self.language]:
					#self.window_cubeFinder(disparity)
					try:
						self.disparity, value_disparity = self.deepMap_updater(imageToDisp, values)
						self.autoFinderWindow.run(self.disparity)
					except:
						self.window.FindElement("_output_").Update('')
						print("ERROR:Firstly create disparity map!")
						print(traceback.format_exc()) 

				#########----------------------------------------
				# События рисования на графе нажатием кнопок мыши

				# Нажали на кнопку мыши первый раз
				# Алгоритм запомнит точку и будет ждать вторую
				if event == "graph":
					x,y = values["graph"]
					if not self.dragging:
						self.start_p = (x,y)
						self.dragging = True
						print(f"start z = {self.Window3D.pointcloud[int((abs(y-cfv.WINDOW_HEIGHT))*cfv.IMAGE_HEIGHT/cfv.WINDOW_HEIGHT)][int(x*cfv.IMAGE_WIDTH/cfv.WINDOW_WIDTH)][2]}")
						print(f"start disparity = {self.disparity_value[int((abs(y-cfv.WINDOW_HEIGHT))*cfv.IMAGE_HEIGHT/cfv.WINDOW_HEIGHT)][int(x*cfv.IMAGE_WIDTH/cfv.WINDOW_WIDTH)]}")
					else:
						self.end_p = (x,y)
						print(f"end z = {self.Window3D.pointcloud[int((abs(y-cfv.WINDOW_HEIGHT))*cfv.IMAGE_HEIGHT/cfv.WINDOW_HEIGHT)][int(x*cfv.IMAGE_WIDTH/cfv.WINDOW_WIDTH)][2]}")
						print(f"end disparity = {self.disparity_value[int((abs(y-cfv.WINDOW_HEIGHT))*cfv.IMAGE_HEIGHT/cfv.WINDOW_HEIGHT)][int(x*cfv.IMAGE_WIDTH/cfv.WINDOW_WIDTH)]}")

				# Если мы нажали на кнопку мыши второй раз
				elif event.endswith('+UP') and self.end_p!= None:
					self.start_point = self.start_p
					self.end_point = self.end_p
					print(f"grabbed rectangle from {self.start_point} to {self.end_point}")
					if self.start_p != None or self.end_p != None or [self.start_p, self.end_p] != None:
						self.lines.append([[int(self.start_p[0]*cfv.IMAGE_WIDTH/cfv.WINDOW_WIDTH), int((abs(self.start_p[1]-cfv.WINDOW_HEIGHT))*cfv.IMAGE_HEIGHT/cfv.WINDOW_HEIGHT)],\
							[int(self.end_p[0]*cfv.IMAGE_WIDTH/cfv.WINDOW_WIDTH), int((abs(self.end_p[1]-cfv.WINDOW_HEIGHT))*cfv.IMAGE_HEIGHT/cfv.WINDOW_HEIGHT)]])
	
						# Производятся граничные условия для значений координат
						self.lines[-1][0] = self.boundary_condition(self.lines[-1][0])
						self.lines[-1][1] = self.boundary_condition(self.lines[-1][1])
						
						# Для рисования линий	
						self.lines_for_view.append([self.start_p, self.end_p])
					# Обнуляем переменные
					self.start_p, self.end_p = None, None
					self.dragging = False
				
				# Создаем карту несоответсвий и отрисовываем ее в правой части
				if event == self.create_map[self.language]:
					self.disparity, self.autoFinderWindow.disparity_value = self.deepMap_updater(imageToDisp, values)
					self.disparity_value = self.autoFinderWindow.disparity_value
					self.Window3D.updatePointCloud(self.disparity,self.disparity_value, self.ifCamPi)
					#self.window.FindElement("_output_").Update('')
					print("Deep map is created.")

				# Нажатие на кнопку "Вычислить", которая должна вернуть 
				# наименования граней и их размеры(Для линий которые мы сами нарисовали).
				if event == self.find_distances[self.language]:
					try:
						self.findAllDistances()
					except:
						print("'find distances' dont work")
						print(traceback.format_exc())

			except IndexError:
				print(traceback.format_exc())

			return True

		# В зависимости от того, есть ли расбериПи или нет - мы запускаем соответсвующую опцию
		if self.ifCamPi:
			# The PSG "Event Loop"
			for frame in self.camera.capture_continuous(self.capture, format="bgra", use_video_port=True, resize=(self.cam_width,self.cam_height)): 
				loop_flag = main_loop(frame) 
				if not loop_flag:
					break
		else:
			while True:
				# Двойное изображение
				imageToDisp = self.imageToDisp[self.index]
				loop_flag = main_loop(imageToDisp) 
				if not loop_flag:
					break

	# Функция, которая выводит информацию о всех
	# найденных и нарисованных линиях
	def findAllDistances(self) -> None:
		# Проверяем есть ли линии
		if len(self.lines) == 0 and len(self.autoFinderWindow.auto_lines) == 0:
			self.window.FindElement("_output_").Update('')
			print("No lines that I can find.")
		else:
			self.window.FindElement("_output_").Update('')

			# Отрисовываем линии найденные автоопределителем 
			mul_coef = 1
			k=0
			for i in range(0, len(self.autoFinderWindow.auto_lines), 4):
				print(f"Object {int(i/4) + 1}")
				for j in range(4):
					box0,box1 = self.autoFinderWindow.auto_lines[k+j]
					print(f"{self.letter_dict[k]*mul_coef}{j+1} : {round(self.old_determine_line([box0,box1]), 2)} mm")
				if self.letter_dict[k] == 'Z':
					k=0
					mul_coef = mul_coef + 1
					continue
				k = k+1		
			
			# Отрисовываем нарисованные линии
			for i,line in enumerate(self.lines):
				if i==0:
					print("\nLines:")
				line_size = self.determine_line(line)
				print(f"{self.letter_dict[i]} : {round(line_size,2)} mm")
		
	def deepMap_updater(self,imageToDisp, values):
		imgL, imgR = calibrate_two_images(imageToDisp, self.ifCamPi)
		rectified_pair = (imgL, imgR)	
		self.db.update_mWinParameters(values)
		return stereo_depth_map(rectified_pair, self.db.mWinParameters)

	# Функция, находящее расстояние между двумя точками в псевдо-3д просстранстве
	def old_determine_line(self, line, baseline=0.065, focal=1442, rescale=1)-> float:
		A = self.boundary_condition(line[0])
		B = self.boundary_condition(line[1])

		disp1 = self.disparity_value[line[0][1]][line[0][0]]
		disp2 = self.disparity_value[line[1][1]][line[1][0]]
		depth1 = baseline *focal/ (rescale * disp1)
		depth2 = baseline *focal/ (rescale * disp2)

		line_size = abs(math.sqrt(pow(B[0] - A[0], 2) + pow(B[1] - A[1],2) + pow(depth2 - depth1, 2)))/2.65
		
		# if line_size>33 and line_size < 56:
		# 	if line_size < 40:
		# 		if line_size >35:
		# 			line_size = line_size + 5
		# 		else:
		# 			line_size = line_size + 10
		# 	elif line_size >45:
		# 		if line_size<50:
		# 			line_size = line_size -5
		# 		else:
		# 			line_size = line_size -10

		return line_size

	# Граничные условия для координат линий
	@staticmethod
	def boundary_condition(A):
		if A[1]>=cfv.IMAGE_HEIGHT:
			A[1] = cfv.IMAGE_HEIGHT - 1
		if A[0] >= cfv.IMAGE_WIDTH:
			A[0]=cfv.IMAGE_WIDTH - 1
		return A
		
	# Функция, которая находит расстояние между двумя точками из 3D облака точек
	def determine_line(self, line, baseline=0.065, focal=1442, rescale=1)-> float:
		A = self.boundary_condition(line[0])
		B = self.boundary_condition(line[1])

		points = self.Window3D.pointcloud
		Xa,Ya,Za = points[A[1]][A[0]]
		Xb,Yb,Zb = points[B[1]][B[0]]

		# print(f"point1 = ({Xa}, {Ya},{Za}) ; point2 = ({Xb}, {Yb},{Zb})")
		line_size = abs(math.sqrt(pow(Xb - Xa, 2)+pow(Yb - Ya, 2)+pow(Zb - Za, 2)))*23.46-8

		return line_size
	
	# Просто находит расстояние между двумя точками на плоскости
	def straight_determine_line(self, line) -> float:
		A = line[0]
		B = line[1]
		
		return abs(math.sqrt(pow((B[0] - A[0])/8.0, 2) + pow((B[1] - A[1])/7.6,2))*1/3)
	
	# Function that find distance on pixel analisys around points
	def smart_determine_line(self, line) -> float:
		A = self.boundary_condition(line[0])
		B = self.boundary_condition(line[1])
		#print(f"A = {A}")
		
		value = 0
		# for A
		for p in [A,B]:
			disparity = self.disparity_value[p[1]-10:p[1]+10, p[0]-10:p[0]+10]
			
			value = np.quantile(disparity, [0.5])
			#print(value)
			
			index_array = np.where(np.isclose(disparity, value, 0.1))
			i = index_array[0][0]
			j = index_array[1][0]
			
			p = [i,j]
		
		# ~ print(f"point0 = {A}, point1 = {B}")
		# ~ print(f"value is {value}")
		# ~ print(f"disp is = {self.autoFinderWindow.disparity_value[A[1]][A[0]]}")
		# ~ print(f"disp is = {self.autoFinderWindow.disparity_value[B[1]][B[0]]}")

		points = self.Window3D.pointcloud
		Xa,Ya,Za = points[A[1]][A[0]]
		Xb,Yb,Zb = points[B[1]][B[0]]
		line_size = abs(math.sqrt(pow(Xb - Xa, 2)+pow(Yb - Ya, 2)+pow(Zb - Za, 2)))*23.46
		
		# if line_size>33 and line_size < 68:
		# 	if line_size < 40:
		# 		if line_size >35:
		# 			line_size = line_size + 5
		# 		else:
		# 			line_size = line_size + 10
		# 	elif line_size >60:
		# 		if line_size<55:
		# 			line_size = line_size -15
		# 		else:
		# 			line_size = line_size -20

		return line_size
		

	# Рисуем на графе
	def drawStraightLine(self) -> None:
		################################################
		# Рисовательная часть
		#--------------------------------------
		# Перерисовка графа
		if self.a_id:
			self.graph.DeleteFigure(self.a_id)
		self.disparity_to_show = cv2.resize(self.disparity, dsize=(cfv.WINDOW_WIDTH, cfv.WINDOW_HEIGHT), interpolation = cv2.INTER_CUBIC)
		self.a_id = self.graph.DrawImage(data=cv2.imencode('.png', self.disparity_to_show)[1].tobytes(), location=(0, cfv.WINDOW_HEIGHT))
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
		###############################################

	def close(self):
		self.window.close()

