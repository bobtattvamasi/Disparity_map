import cv2, PySimpleGUI as sg
from determine_object_helper import convert_to_bytes, calibrate_two_images, stereo_depth_map, contours_finder
import base64
import glob
import sys, traceback
from point_cloud import stereo_depth_map as depth_map, create_points_cloud
from abc import ABC, abstractmethod
import subprocess


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
		self.imageToDisp, self.index = self.image_to_disp('./data/scenes4test/*.png')

		# --------------------------------------------
		# Разделяем пространство окна на левое и правое
		#
		# Левая колонка:
		left_column = [
			# Место под левое изображение
			[self.sg.Image(filename='', key='image')],
			# Далее идут слайдеры параметров для настройки выходной карты глубин
			[self.sg.Text("SpklWinSze"), self.sg.Slider(range=(0, 300), orientation='h', size=(34, 10), default_value=100)],
			[self.sg.Text("SpcklRng"), self.sg.Slider(range=(0, 40), orientation='h', size=(34, 10), default_value=15)],
			[self.sg.Text("UnicRatio"), self.sg.Slider(range=(1, 80), orientation='h', size=(34, 10), default_value=10)],
			[self.sg.Text("TxtrThrshld"), self.sg.Slider(range=(0, 1000), orientation='h', size=(34, 10), default_value=100)],
			[self.sg.Text("NumOfDisp"), self.sg.Slider(range=(16, 256), orientation='h', size=(34, 10), default_value=128)],
			[self.sg.Text("MinDISP"), self.sg.Slider(range=(-300, 300), orientation='h', size=(34, 10), default_value=61)],
			[self.sg.Text("PreFiltCap"), self.sg.Slider(range=(5, 63), orientation='h', size=(34, 10), default_value=29)],
			[self.sg.Text("PFS"), self.sg.Slider(range=(5, 255), orientation='h', size=(34, 10), default_value=5)],
			[self.sg.Text("SWS"), self.sg.Slider(range=(5, 255), orientation='h', size=(34, 10), default_value=5)],
			# Кнопка по которой вычисляются размеры найденных граней
			[self.sg.Button('Determine', size=(15,2)), 
			# Кнопка котороя показывает следующее изображение
			self.sg.Button('Next Picture ->', size=(15,2))]
		]

		# Правая колонка:
		#
		right_column = [
			# Место для отображения второй картинки
			# Только в виде графа, что бы можно было 
			# рисовать поверх грани
			[self.sg.Graph(canvas_size=(700,400),
					graph_bottom_left=(0, 0),
					graph_top_right=(700, 400),
					key="graph",
					change_submits=True, # mouse click events
					drag_submits=True # mouse drag events]
					)],
			# Вывод всей важной информации происходит здесь
			[self.sg.Output(size=(64, 12))],
			[self.sg.Text("minVal"), self.sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=50)],
			[self.sg.Text("maxVal"), self.sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=200)],
			[self.sg.Text("Layer"), self.sg.Slider(range=(0, 7), orientation='h', size=(34, 10), default_value=0)]
		]

		# Объединяемвсе в один объект
		layout  = [[self.sg.Column(left_column, element_justification='c'), 
					self.sg.VSeperator(),
					self.sg.Column(right_column, element_justification='c')]
				]
		# Наше окно
		self.window = self.sg.Window(TextForApp, 
						layout, icon=self.icon, resizable=True)

	# Основная функция в котором работает наше окно
	def run(self):
		# Двойное изображение
		imageToDisp = self.imageToDisp[self.index]

		# Создаем словарь параметров для настройки карты глубины,
		# коорые ниже можно будет регулировать через ползунки.
		parameters = {'SpklWinSze':0,
				'SpcklRng': 0,
				'UnicRatio':25,
				'TxtrThrshld':0,
				'NumOfDisp':80,
				'MinDISP':61,
				'PreFiltCap':5,
				'PFS': 5,
				'SWS': 5}

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

		# The PSG "Event Loop"
		while True:                     
			event, values = self.window.Read(timeout=20, timeout_key='timeout')      # get events for the window with 20ms max wait

			if event is None or event == self.sg.WIN_CLOSED or event == 'Cancel':  
				break  

			try:
				parameters['SpklWinSze'] = int(values[0])
				parameters['SpcklRng'] = int(values[1])
				parameters['UnicRatio'] = int(values[2])
				parameters['TxtrThrshld'] = int(values[3])
				parameters['NumOfDisp'] = int(values[4]/16)*16
				parameters['MinDISP'] = int(values[5])
				parameters['PreFiltCap'] = int(values[6]/2)*2+1
				parameters['PFS'] = int(values[7]/2)*2+1
				parameters['SWS'] = int(values[8]/2)*2+1

				minVal = int(values[10])
				maxVal = int(values[11])
				layer  = int(values[12])

				# Получение карты глубины
				disparity = stereo_depth_map(rectified_pair, parameters)
				# Нахождение на карте глубин границ и отображение их
				disparity = contours_finder(disparity, minVal, maxVal, layer, -1)
				
				#
				# Рисовательная часть(возможно не нужно)
				#--------------------------------------
				# Перерисовка графа
				if a_id:
					graph.DeleteFigure(a_id)
				a_id = graph.DrawImage(data=cv2.imencode('.png', disparity)[1].tobytes(), location=(0, 400))
				# Рисовние линий
				if start_point is not None and end_point is not None:
					graph.DrawLine(start_point, end_point, color='purple', width=10)
				graph.TKCanvas.tag_lower(a_id)
				#--------------------------------------
				# Конец рисовательной части
				#

				# Update image in window
				self.window.FindElement('image').Update(data=cv2.imencode('.png', rectified_pair[0])[1].tobytes())

			except IndexError:
				print(traceback.format_exc()) 


			# Обработчики событий
			# ---------------------

			# Нажатие на кнопку "Следующая картинка"
			if event == "Next Picture ->":
				self.index = self.index + 1
				if self.index >= len(self.imageToDisp):
					self.index=0
				imageToDisp = self.imageToDisp[self.index] 
				imgL, imgR = calibrate_two_images(imageToDisp)
				rectified_pair = (imgL, imgR)

			# События рисования на графе нажатием кнопки мыши
			if event == "graph":
				x,y = values["graph"]
				#print (f"mouse down at ({x},{y})")
				if not dragging:
					start_p = (x,y)
					dragging = True
				else:
					end_p = (x,y)
					
			elif event.endswith('+UP'):
				start_point = start_p
				end_point = end_p
				print(f"grabbed rectangle from {start_point} to {end_point}")
				start_p, end_p = None, None
				dragging = False

			# Нажатие на кнопку "Вычислить", которая должна вернуть 
			# наименования граней и их размеры.
			if event is 'Determine':
				# Вычисляется облако точек
				disparity2, points_3, colors = create_points_cloud(imageToDisp, parameters)
				print(f'Info: Filepaths correctly defined.{values[0]}')  

	def close(self):
		self.window.close()