import os.path
import PIL.Image
import io
import base64

import cv2
import numpy as np
import imutils
# from death_map import *

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
def calibrate_two_images(imageToDisp, photoDim=None, imageDim=None):
	photo_width = 3840
	photo_height = 1088
	image_width = 1920
	image_height = 1088

	image_size = (image_width,image_height)
	pair_img = cv2.imread(imageToDisp,0)
	print('Read and split image...')
	imgLeft = pair_img [0:photo_height,0:image_width] #Y+H and X+W
	imgRight = pair_img [0:photo_height,image_width:photo_width] #Y+H and X+W

	# Implementing calibration data
	print('Read calibration data and rectifying stereo pair...')

	try:
		npzfile = np.load('./data/calibration_data/{}p/stereo_camera_calibration.npz'.format(image_height))
	except:
		print("Camera calibration data not found in cache, file " & './data/calibration_data/{}p/stereo_camera_calibration.npz'.format(480))
		exit(0)

	imageSize = tuple(npzfile['imageSize'])
	leftMapX = npzfile['leftMapX']
	leftMapY = npzfile['leftMapY']
	rightMapX = npzfile['rightMapX']
	rightMapY = npzfile['rightMapY']

	width_left, height_left = imgLeft.shape[:2]
	width_right, height_right = imgRight.shape[:2]

	if 0 in [width_left, height_left, width_right, height_right]:
		print("Error: Can't remap image.")

	imgL = cv2.remap(imgLeft, leftMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
	imgR = cv2.remap(imgRight, rightMapX, rightMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

	imgRs = cv2.resize (imgR, dsize=(640, 362), interpolation = cv2.INTER_CUBIC)
	imgLs = cv2.resize (imgL, dsize=(640, 362), interpolation = cv2.INTER_CUBIC)

	# cv2.imshow("right", imgRs)
	# cv2.imshow("left", imgLs)
	cv2.waitKey(0)
	return imgLs, imgRs

# Функция возвращает карту шлубин по калиброванным стереофото
def stereo_depth_map(rectified_pair, parameters, ifsave=True):
	SPWS = parameters['SpklWinSze']
	PFS = parameters['PFS']
	PFC = parameters['PreFiltCap']
	MDS = parameters['MinDISP']
	NOD = parameters['NumOfDisp']
	TTH = parameters['TxtrThrshld']
	UR = parameters['UnicRatio']
	SR = parameters['SpcklRng']
	SWS = parameters['SWS']

	c, r = rectified_pair[0].shape
	disparity = np.zeros((c, r), np.uint8)
	sbm = cv2.StereoBM_create(numDisparities=16, blockSize=15)
	sbm.setPreFilterType(1)
	sbm.setPreFilterSize(PFS)
	sbm.setPreFilterCap(PFC)
	sbm.setMinDisparity(MDS)
	sbm.setNumDisparities(NOD)
	sbm.setTextureThreshold(TTH)
	sbm.setUniquenessRatio(UR)
	sbm.setSpeckleRange(SR)
	sbm.setSpeckleWindowSize(SPWS)
	dmLeft = rectified_pair[0]
	dmRight = rectified_pair[1]
	disparity = sbm.compute(dmLeft, dmRight)
	#disparity_visual = cv.CreateMat(c, r, cv.CV_8U)
	local_max = disparity.max()
	local_min = disparity.min()
	# print ("MAX " + str(local_max))
	# print ("MIN " + str(local_min))
	disparity_visual = (disparity-local_min)*(1.0/(local_max-local_min))
	if ifsave:
		disparity_grayscale = (disparity-local_min)*(65535.0/(local_max-local_min))
		#disparity_grayscale = (disparity+208)*(65535.0/1000.0) # test for jumping colors prevention 
		disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
		disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
		#cv2.imshow("Image", disparity_color)
		
		cv2.imwrite("fulldisp.jpg",disparity_color)
	local_max = disparity_visual.max()
	local_min = disparity_visual.min()
	# print ("MAX " + str(local_max))
	# print ("MIN " + str(local_min))
	#cv.Normalize(disparity, disparity_visual, 0, 255, cv.CV_MINMAX)
	#disparity_visual = np.array(disparity_visual)
	disparity_visual = cv2.resize (disparity_visual, dsize=(960, 543), interpolation = cv2.INTER_CUBIC)
	return disparity_color

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
	img= cv2.drawContours(img, screenCnt, layer, (0, 255,0), 2, cv2.LINE_AA, hierarchy, 0)

	# cv2.imshow("edges", edges)
	# cv2.imshow("img", img)
	# cv2.waitKey(0)
	return img









