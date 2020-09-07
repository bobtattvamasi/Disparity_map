import cv2
import numpy as np
import imutils

from GUI.baseInterface import baseInterface
from db.DBtools import *
from tools.determine_object_helper import autoFindRect


class autoDetectRectWindow(baseInterface):
	def __init__(self, themeStyle, TextForApp):
		super().__init__(themeStyle, TextForApp)

		self.auto_lines = []
		self.secondWin_parameters = read_csv("db/secondWin.csv")

		
		
		self.TextForApp = TextForApp

		

	def run(self, image):
		left_column = [[self.sg.Image(filename='', key='cube_image'), self.sg.Image(filename='', key='another_image')],
					[self.sg.Text("lowH"), self.sg.Slider(range=(0, 179), orientation='h', size=(34, 10), default_value=self.secondWin_parameters["lowH"])],
					[self.sg.Text("highH"), self.sg.Slider(range=(0, 179), orientation='h', size=(34, 10), default_value=self.secondWin_parameters["highH"])],
					[self.sg.Text("lowS"), self.sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=self.secondWin_parameters["lowS"])],
					[self.sg.Text("highS"), self.sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=self.secondWin_parameters["highS"])],
					[self.sg.Text("lowV"), self.sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=self.secondWin_parameters["lowV"])],
					[self.sg.Text("highV"), self.sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=self.secondWin_parameters["highV"])],
			[self.sg.Text("minVal"), self.sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=50)],
			[self.sg.Text("maxVal"), self.sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=200)],
			[self.sg.Text("Layer"), self.sg.Slider(range=(0, 7), orientation='h', size=(34, 10), default_value=0)],
				[self.sg.Text("ThMin"), self.sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=240)],
				[self.sg.Text("ThMax"), self.sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=255)],
				[#self.sg.Button('find contours', size=(15,2)),
				self.sg.Button('save settings', size=(15,2))]
					]
		right_column = [[self.sg.Image(filename='', key='mask_contor_image')],
						[self.sg.Image(filename='', key='hsv_image')]
						]
		layout  = [[self.sg.Column(left_column, element_justification='c'), 
					self.sg.VSeperator(),
					self.sg.Column(right_column, element_justification='c')]
					]
		window = self.sg.Window(self.TextForApp, layout)
		self.auto_lines = []
		hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		# second_inRange = cv2.inRange(image, np.array([0,0,0]), np.array([16,50,255]))
		
		while True:
			event, values = window.read(timeout=200)
			if not event:
				break
			if event == self.sg.WIN_CLOSED or event == 'Cancel':
				break



			self.secondWin_parameters["lowH"] = int(values[0])
			self.secondWin_parameters["highH"] = int(values[1])
			self.secondWin_parameters["lowS"] = int(values[2])
			self.secondWin_parameters["highS"] = int(values[3])
			self.secondWin_parameters["lowV"] = int(values[4])
			self.secondWin_parameters["highV"] = int(values[5])

			self.secondWin_parameters["minVal"] = int(values[6])
			self.secondWin_parameters["maxVal"] = int(values[7])
			self.secondWin_parameters["layer"] = int(values[8])
			self.secondWin_parameters["Thmin"] = int(values[9])
			self.secondWin_parameters["Thmax"] = int(values[10])

			_, mask_frame, thresh = autoFindRect(image, hsv_frame, self)

			
			if event == 'save settings':
				save_csv("db/secondWin.csv", self.secondWin_parameters)
			
			image_to_show = cv2.resize(image, dsize=(640, 362), interpolation = cv2.INTER_CUBIC)
			thresh_to_show = cv2.resize(thresh, dsize=(640, 362), interpolation = cv2.INTER_CUBIC)
			mask_frame_to_show = cv2.resize(mask_frame, dsize=(640, 362), interpolation = cv2.INTER_CUBIC)

			
			window.FindElement('cube_image').Update(data=cv2.imencode('.png', image_to_show)[1].tobytes())
			window.FindElement('mask_contor_image').Update(data=cv2.imencode('.png', thresh_to_show)[1].tobytes())
			window.FindElement('hsv_image').Update(data=cv2.imencode('.png', mask_frame_to_show)[1].tobytes())

		window.close()

	# def close(self):
	# 	window.close()



