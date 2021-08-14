import cv2
import numpy as np
import imutils

from GUI.BaseWindow import BaseWindow
from db.DBtools import *
from tools.ImageProccessHelper import autoFindRect
from config.config import configValues as cfv
from tools.VertexFinder import VertexFinder


class AutoDetectRectWindow(BaseWindow):

	def __init__(self, themeStyle, TextForApp):
		super().__init__(themeStyle, TextForApp)

		self.auto_lines = []
		self.TextForApp = TextForApp

		self.settings = ('Settings', 'Настройки')
		self.save_settings = ('save settings', 'сохранить')
		self.disparity_value = None

		self.left_image = None
		self.VertexFinder = VertexFinder()

	def run(self, image) -> None:
		left_column = [[self.sg.Image(filename='', key='cube_image'), self.sg.Image(filename='', key='another_image')],
				[self.sg.Frame(self.settings[self.language],[
					[self.sg.Text("lowH"), self.sg.Slider(range=(0, 179), orientation='h', size=(34, 10), default_value=self.db.aDRWinParameters["lowH"])],
					[self.sg.Text("highH"), self.sg.Slider(range=(0, 179), orientation='h', size=(34, 10), default_value=self.db.aDRWinParameters["highH"])],
					[self.sg.Text("lowS"), self.sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=self.db.aDRWinParameters["lowS"])],
					[self.sg.Text("highS"), self.sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=self.db.aDRWinParameters["highS"])],
					[self.sg.Text("lowV"), self.sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=self.db.aDRWinParameters["lowV"])],
					[self.sg.Text("highV"), self.sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=self.db.aDRWinParameters["highV"])],
			[self.sg.Text("minVal"), self.sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=50)],
			[self.sg.Text("maxVal"), self.sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=200)],
			[self.sg.Text("Layer"), self.sg.Slider(range=(0, 7), orientation='h', size=(34, 10), default_value=0)],
				[self.sg.Text("ThMin"), self.sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=240)],
				[self.sg.Text("ThMax"), self.sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=255)],
				[self.sg.Button(self.save_settings[self.language], size=(10,1))]
				])]
					]
		right_column = [[self.sg.Image(filename='', key='mask_contor_image')],
						[self.sg.Image(filename='', key='hsv_image')]
						]
		layout  = [[self.sg.Column(left_column), 
					self.sg.VSeperator(),
					self.sg.Column(right_column, element_justification='c')]
					]
		window = self.sg.Window(self.TextForApp, layout)

		# Переменная для хранения линий
		# с автоопределения прямоугольников
		self.auto_lines = []
		hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		
		while True:
			event, values = window.read(timeout=200)
			if not event:
				break
			if event == self.sg.WIN_CLOSED or event == 'Cancel':
				break

			self.db.update_aDRWinParameters(values)

			mask_frame, thresh = autoFindRect(image, hsv_frame, self)

			
			if event == self.save_settings[self.language]:
				self.db.save_csv("db/secondWin.csv", self.db.aDRWinParameters)
			
			image_to_show = cv2.resize(image, dsize=(cfv.WINDOW_WIDTH, cfv.WINDOW_HEIGHT), interpolation = cv2.INTER_CUBIC)
			thresh_to_show = cv2.resize(thresh, dsize=(cfv.WINDOW_WIDTH, cfv.WINDOW_HEIGHT), interpolation = cv2.INTER_CUBIC)
			mask_frame_to_show = cv2.resize(mask_frame, dsize=(cfv.WINDOW_WIDTH, cfv.WINDOW_HEIGHT), interpolation = cv2.INTER_CUBIC)

			
			window['cube_image'].Update(data=cv2.imencode('.png', image_to_show)[1].tobytes())
			# Результат действия наложения цветовой маски (Правое верхнее изображение)
			window['mask_contor_image'].Update(data=cv2.imencode('.png', thresh_to_show)[1].tobytes())
			#
			window['hsv_image'].Update(data=cv2.imencode('.png', mask_frame_to_show)[1].tobytes())

		window.close()




