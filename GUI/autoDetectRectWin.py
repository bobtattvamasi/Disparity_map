import cv2
import numpy as np
import imutils

from GUI.baseInterface import baseInterface
from GUI.tools_for_intaface import *
from determine_object_helper import autoFindRect


class autoDetectRectWin(baseInterface):
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

			# hsv_min = np.array((self.secondWin_parameters["lowH"], self.secondWin_parameters["lowS"], self.secondWin_parameters["lowV"]), np.uint8)
			# hsv_max = np.array((self.secondWin_parameters["highH"], self.secondWin_parameters["highS"], self.secondWin_parameters["highV"]), np.uint8)

			

			# #mask_frame = contours_finder(mask_frame, minVal, maxVal, layer, -1)

			# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			# height, width = gray.shape
			# # ~ mask = np.zeros((height,width))
			# # ~ cv2.rectangle(mask, (100,60), (460,355), 255, -1)
			# # ~ #masked_data = cv2.bitwise_and(gray, gray, mask=mask)
			
			# # ~ #output = cv2.bitwise_and(image, image, mask = second_inRange)
			# # ~ #output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
			
			# # ~ gray[mask < 255] = 0
			# # ~ hsv_frame[mask < 255] = 0
			# mask_frame = cv2.inRange(hsv_frame, hsv_min, hsv_max)
			# #img = image[140:355, 80:460]
			
			# #gray = image
			# blurred = cv2.GaussianBlur(mask_frame, (5, 5), 0)
			
			# thresh = cv2.threshold(blurred, self.secondWin_parameters["Thmin"], 
			# 	self.secondWin_parameters["Thmax"], cv2.THRESH_BINARY)[1]
				
			# image_blur = cv2.medianBlur(thresh, 25)
			
			# image_res, thresh = cv2.threshold(image_blur, 240,255, cv2.THRESH_BINARY_INV)
			
			# #---find black on white
			

			# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			# cv2.CHAIN_APPROX_SIMPLE)
			# cnts = imutils.grab_contours(cnts)
			# ratio =1
			# boxes = []
			if event == 'save settings':
				save_csv("db/secondWin.csv", self.secondWin_parameters)

			#img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
			# cX = 0
			# cY = 0
			# for i,c in enumerate(cnts):
			# 	# compute the center of the contour, then detect the name of the
			# 	# shape using only the contour
			# 	M = cv2.moments(c)
			# 	if M["m00"] != 0:
			# 		cX = int((M["m10"] / M["m00"]) * ratio)
			# 		cY = int((M["m01"] / M["m00"]) * ratio)
			# 	# multiply the contour (x, y)-coordinates by the resize ratio,
			# 	# then draw the contours and the name of the shape on the image
			# 	c = c.astype("float")
			# 	c *= ratio
			# 	c = c.astype("int")
			# 	contour_area = cv2.contourArea(c)
			# 	if contour_area < 150:
			# 		continue
			# 	#print(f"{i} : area = {contour_area}")
			# 	peri = cv2.arcLength(c, True)
			# 	c = cv2.approxPolyDP(c, 0.020 * peri, True)
			# 	# rotated rect constructor
			# 	rect = cv2.minAreaRect(c)
				
			# 	box = np.int0(cv2.boxPoints(rect))
			# 	#print(f"recr = {box}")
			# 	#-------------------------
				
			# 	self.auto_lines.append([box[0],box[1]])
			# 	self.auto_lines.append([box[1],box[2]])
			# 	self.auto_lines.append([box[2],box[3]])
			# 	self.auto_lines.append([box[3],box[0]])
				
			# 	# # rect constructor ------
			# 	# (x,y,w,h) = cv2.boundingRect(c)
			# 	# boxes.append([x,y,x+w, y+h])
			# 	# cv2.rectangle(image, (x,y), (x+w,y+h),(0,0,255),2)
			# 	# -----------------------
			# 	image = cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
			
				

			window.FindElement('cube_image').Update(data=cv2.imencode('.png', image)[1].tobytes())
			window.FindElement('mask_contor_image').Update(data=cv2.imencode('.png', thresh)[1].tobytes())
			window.FindElement('hsv_image').Update(data=cv2.imencode('.png', mask_frame)[1].tobytes())

		window.close()

	# def close(self):
	# 	window.close()



