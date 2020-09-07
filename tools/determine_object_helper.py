import os.path
import PIL.Image
import io
import base64

import cv2
import numpy as np
#import imutils
# from death_map import *
from config.config import configValues
import imutils

# Функция для отображения картинок на фронте
def convert_to_bytes(file_or_bytes, resize=None):
	'''
	Will convert into bytes and optionally resize an image that is a file or a base64 bytes object.
	Turns into  PNG format in the process so that can be displayed by tkinter
	:param file_or_bytes: either a string filename or a bytes base64 image object
	:type file_or_bytes:  (Union[str, bytes])
	:param resize:  optional new size
	:type resize: (Tuple[int, int] or None)
	:return: (bytes) a byte-string object
	:rtype: (bytes)
	'''
	if isinstance(file_or_bytes, str):
		img = PIL.Image.open(file_or_bytes)
	else:
		try:
			img = PIL.Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))
		except Exception as e:
			dataBytesIO = io.BytesIO(file_or_bytes)
			img = PIL.Image.open(dataBytesIO)

	cur_width, cur_height = img.size
	if resize:
		new_width, new_height = resize
		scale = min(new_height/cur_height, new_width/cur_width)
		img = img.resize((int(cur_width*scale), int(cur_height*scale)), PIL.Image.ANTIALIAS)
	bio = io.BytesIO()
	img.save(bio, format="PNG")
	del img
	return bio.getvalue()

# Функция калибрует два изображения, разделяет их и эти два возвращает
def calibrate_two_images(imageToDisp, ifCamPi = True, photoDim=None, imageDim=None):
	# Определяем размеры(ширина,высота) сдвоенного изображения со стереокамеры и
	# разделенных левого и правого.
	photo_width = configValues.PHOTO_WIDTH
	photo_height = configValues.PHOTO_HEIGHT
	image_width = configValues.IMAGE_WIDTH
	image_height = configValues.IMAGE_HEIGHT
	image_size = (image_width,image_height)

	# Cчитываем стерео-изображение
	# if we from PiCamera
	pair_img = None
	if ifCamPi:
		pair_img = cv2.cvtColor(imageToDisp,cv2.COLOR_BGR2GRAY)
	# If we from pictures
	else:
		pair_img = cv2.imread(imageToDisp,0)

	# Разделяем на левое и правое
	imgLeft = pair_img [0:photo_height,0:image_width] #Y+H and X+W
	imgRight = pair_img [0:photo_height,image_width:photo_width] #Y+H and X+W

	# Загружаем калибровочные данные
	try:
		npzfile = np.load('./data/calibration_data/{}p/stereo_camera_calibration.npz'.format(image_height))
	except:
		print("Camera calibration data not found in cache, file " & './data/calibration_data/{}p/stereo_camera_calibration.npz'.format(image_height))
		exit(0)

	# Параметры калибровки
	imageSize = tuple(npzfile['imageSize'])
	leftMapX = npzfile['leftMapX']
	leftMapY = npzfile['leftMapY']
	rightMapX = npzfile['rightMapX']
	rightMapY = npzfile['rightMapY']

	width_left, height_left = imgLeft.shape[:2]
	width_right, height_right = imgRight.shape[:2]

	if 0 in [width_left, height_left, width_right, height_right]:
		print("Error: Can't remap image.")

	# Трансформация левого иправого изображения с помощью матриц калибровки
	imgL = cv2.remap(imgLeft, leftMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
	imgR = cv2.remap(imgRight, rightMapX, rightMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
	return imgL, imgR

def resize_rectified_pair(rectified_pair):
	imgRs = cv2.resize (rectified_pair[1], dsize=(640, 362), interpolation = cv2.INTER_CUBIC)
	imgLs = cv2.resize (rectified_pair[0], dsize=(640, 362), interpolation = cv2.INTER_CUBIC)
	return imgLs, imgRs

autotune_min = 10000000
autotune_max = -10000000

def stereo_depth_map(rectified_pair, parameters):
	dmLeft = rectified_pair[0]
	dmRight = rectified_pair[1]

	#disparity = np.zeros((c,r), np.uint8)
	sbm = cv2.StereoBM_create(numDisparities=16, blockSize=15)
	sbm.setPreFilterType(1)
	sbm.setPreFilterSize(parameters['PFS'])
	sbm.setPreFilterCap(parameters['PreFiltCap'])
	sbm.setMinDisparity(parameters['MinDISP'])
	sbm.setNumDisparities(parameters['NumOfDisp'])
	sbm.setTextureThreshold(parameters['TxtrThrshld'])
	sbm.setUniquenessRatio(parameters['UnicRatio'])
	sbm.setSpeckleRange(parameters['SpcklRng'])
	sbm.setSpeckleWindowSize(parameters['SpklWinSze'])
	
	disparity = sbm.compute(dmLeft, dmRight)
	local_max = disparity.max()
	local_min = disparity.min()

	global autotune_max, autotune_min
	# autotune_max = max(autotune_max, disparity.max())
	# autotune_min = min(autotune_min, disparity.min())
	autotune_max = local_max
	autotune_min = local_min

	disparity_grayscale = (disparity-autotune_min)*(65535.0/(autotune_max-autotune_min))
	disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
	disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
	#print(f"disparity size = {disparity_color.shape}")
	return disparity_color,  disparity.astype(np.float32) / 16.0

# Функция возвращает карту шлубин по калиброванным стереофото
# def stereo_depth_map(rectified_pair, parameters, ifsave=True):
# 	#SPWS = parameters['SpklWinSze']
# 	#PFS = parameters['PFS']
# 	#PFC = parameters['PreFiltCap']
# 	#MDS = parameters['MinDISP']
# 	#NOD = parameters['NumOfDisp']
# 	#TTH = parameters['TxtrThrshld']
# 	#UR = parameters['UnicRatio']
# 	#SR = parameters['SpcklRng']
# 	#SWS = parameters['SWS']
	
# 	#print(f"r,c = {rectified_pair[0].shape}")
# 	c, r = rectified_pair[0].shape
# 	disparity = np.zeros((c,r), np.uint8)
# 	sbm = cv2.StereoBM_create(numDisparities=16, blockSize=15)
# 	sbm.setPreFilterType(1)
# 	sbm.setPreFilterSize(parameters['PFS'])
# 	sbm.setPreFilterCap(parameters['PreFiltCap'])
# 	sbm.setMinDisparity(parameters['MinDISP'])
# 	sbm.setNumDisparities(parameters['NumOfDisp'])
# 	sbm.setTextureThreshold(parameters['TxtrThrshld'])
# 	sbm.setUniquenessRatio(parameters['UnicRatio'])
# 	sbm.setSpeckleRange(parameters['SpcklRng'])
# 	sbm.setSpeckleWindowSize(parameters['SpklWinSze'])
# 	dmLeft = rectified_pair[0]
# 	dmRight = rectified_pair[1]
# 	disparity = sbm.compute(dmLeft, dmRight)
# 	#print(f"disparity.shape = {disparity.shape}")
# 	value_disparity= disparity
# 	#disparity_visual = cv.CreateMat(c, r, cv.CV_8U)
# 	local_max = disparity.max()
# 	local_min = disparity.min()
# 	#print(f"min = {local_min},max = {local_max}")
# 	# print ("MAX " + str(local_max))
# 	# print ("MIN " + str(local_min))
# 	disparity_visual = (disparity-local_min)*(1.0/0.00000001)
# 	if (local_max-local_min) != 0:
# 		disparity_visual = (disparity-local_min)*(1.0/(local_max-local_min))
# 	disparity_grayscale = (disparity-local_min)*(65535.0/0.00000001)
# 	if ifsave:
# 		if (local_max-local_min) != 0:
# 			disparity_grayscale = (disparity-local_min)*(65535.0/(local_max-local_min))
# 		else:
# 			disparity_grayscale = (disparity-local_min)*(65535.0/0.000000000001)
# 		#disparity_grayscale = (disparity+208)*(65535.0/1000.0) # test for jumping colors prevention 
# 		disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
# 		disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
# 		#cv2.imshow("Image", disparity_color)
		
# 		cv2.imwrite("fulldisp.jpg",disparity_color)
# 	local_max = disparity_visual.max()
# 	local_min = disparity_visual.min()
# 	# print ("MAX " + str(local_max))
# 	# print ("MIN " + str(local_min))
# 	#cv.Normalize(disparity, disparity_visual, 0, 255, cv.CV_MINMAX)
# 	#disparity_visual = np.array(disparity_visual)
# 	disparity_visual = cv2.resize (disparity_visual, dsize=(960, 543), interpolation = cv2.INTER_CUBIC)
# 	return disparity_color, value_disparity

# Функция находит границы объекта на карте глубин изображении
def contours_finder(img, minVal, maxVal, layer, index):
	#img = cv2.imread(img_path)
	#img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(img.copy(),minVal, maxVal)
	#cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts, hierarchy = cv2.findContours( edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

	screenCnt = []
	for c in cnts:
		#print(f" CONTOUR_Area = {cv2.contourArea(c)}")
		contour_area = cv2.contourArea(c)
		# approximate the contour
		peri = cv2.arcLength(c, True)
		#print(f"peri={peri}")
		approx = cv2.approxPolyDP(c, 0.020 * peri, True)
		#print(f"spprox = {approx}")
		# if our approximated contour has four points, then
		# we can assume that we have found our screen
		# if contour_area < 25:
		screenCnt.append(approx)
	layer = layer - 1
	#hierarchy = []

	if len(img.shape) == 2 :
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	img= cv2.drawContours(img, screenCnt, -1, (0, 255,0), 2)

	# cv2.imshow("edges", edges)
	# cv2.imshow("img", img)
	# cv2.waitKey(0)
	return img


def autoFindRect(image, hsv_frame, self_, ifPrintRect=False):

	#hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	hsv_min = np.array((self_.secondWin_parameters["lowH"], self_.secondWin_parameters["lowS"], self_.secondWin_parameters["lowV"]), np.uint8)
	hsv_max = np.array((self_.secondWin_parameters["highH"], self_.secondWin_parameters["highS"], self_.secondWin_parameters["highV"]), np.uint8)


	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	height, width = gray.shape

	mask_frame = cv2.inRange(hsv_frame, hsv_min, hsv_max)

	blurred = cv2.GaussianBlur(mask_frame, (5, 5), 0)

	thresh = cv2.threshold(blurred, self_.secondWin_parameters["Thmin"], 
				self_.secondWin_parameters["Thmax"], cv2.THRESH_BINARY)[1]
				
	image_blur = cv2.medianBlur(thresh, 25)
	
	image_res, thresh = cv2.threshold(image_blur, 240,255, cv2.THRESH_BINARY_INV)

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
							cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	ratio = 1

	j=0
	mul_coef = 1

	cX = 0
	cY = 0

	self_.auto_lines = []

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
		if contour_area < 150:
			continue
		#print(f"{i} : area = {contour_area}")
		peri = cv2.arcLength(c, True)
		if peri < 300:
			continue
		c = cv2.approxPolyDP(c, 0.020 * peri, True)
		# rotated rect constructor
		rect = cv2.minAreaRect(c)
		
		box = np.int0(cv2.boxPoints(rect))
		#print(f"recr = {box}")
		#-------------------------
		if ifPrintRect:
			print(f"object {j+1}:")
		# for printing all lines of object
		for k in range(4):
			last = k+1
			if last ==4:
				last = 0			
			cv2.putText(image, str(self_.letter_dict[j]*mul_coef)+str(k+1), (int(box[k][0] + (box[last][0] - box[k][0])/2),int(box[k][1] + (box[last][1] - box[k][1])/2) ), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
			fontScale=1, color=(255,255,255), thickness = 3)
			
			if ifPrintRect:
				self_.secondWindow.auto_lines.append([box[k],box[last]])
				#print(f"{self_.letter_dict[j]*mul_coef}{k+1} : {round(self_.straight_determine_line([box[k],box[last]]), 2)} mm")
				print(f"{self_.letter_dict[j]*mul_coef}{k+1} : {round(self_.old_determine_line(self_.disparity_value, [box[k],box[last]]), 2)} mm")
		# ~ print(f"contou_area = {contour_area}")
		# ~ print(f"perimetr = {peri}")
		j=j+1
		if self_.letter_dict[j] == 'Z':
			j=0
			mul_coef = mul_coef + 1
		
		
			

		if not ifPrintRect:

			self_.auto_lines.append([box[0],box[1]])
			self_.auto_lines.append([box[1],box[2]])
			self_.auto_lines.append([box[2],box[3]])
			self_.auto_lines.append([box[3],box[0]])
		
		# # rect constructor ------
		# (x,y,w,h) = cv2.boundingRect(c)
		# boxes.append([x,y,x+w, y+h])
		# cv2.rectangle(image, (x,y), (x+w,y+h),(0,0,255),2)
		# -----------------------
		image = cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

	return image, mask_frame, thresh














