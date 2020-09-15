import os.path
import PIL.Image
import io
import base64

import cv2
import numpy as np

from config.config import configValues
import imutils

# Функция калибрует два изображения, разделяет их и эти два возвращает
def calibrate_two_images(imageToDisp, ifCamPi = True):
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

# Находит по стереопаре карту несоответсвий
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

	return disparity_color,  disparity.astype(np.float32) / 16.0

# Вырезает найденный прямоугольный 
# кусок c изображения под углом.
def crop_minAreaRect(img, rect):

	# rotate img
	angle = rect[2]
	rows,cols = img.shape[0], img.shape[1]
	M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
	img_rot = cv2.warpAffine(img,M,(cols,rows))

	# rotate bounding box
	rect0 = (rect[0], rect[1], 0.0) 
	box = cv2.boxPoints(rect0)
	pts = np.int0(cv2.transform(np.array([box]), M))[0]    
	pts[pts < 0] = 0

	# crop
	img_crop = img_rot[pts[1][1]:pts[0][1], 
					   pts[1][0]:pts[2][0]]

	print(f"shape_shape = {img_crop.shape}")
	print( "MAX1 = ", max(map(max, img_crop)))
	print("MAX2 = ", img_crop.max())

	return img_crop

# Автоопределение прямоугольных обектов на изображении
def autoFindRect(image, hsv_frame, self_, ifPrintRect=False):

	#hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	hsv_min = np.array((self_.db.aDRWinParameters["lowH"], self_.db.aDRWinParameters["lowS"], self_.db.aDRWinParameters["lowV"]), np.uint8)
	hsv_max = np.array((self_.db.aDRWinParameters["highH"], self_.db.aDRWinParameters["highS"], self_.db.aDRWinParameters["highV"]), np.uint8)


	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	height, width = gray.shape

	mask_frame = cv2.inRange(hsv_frame, hsv_min, hsv_max)

	blurred = cv2.GaussianBlur(mask_frame, (5, 5), 0)

	thresh = cv2.threshold(blurred, self_.db.aDRWinParameters["Thmin"], 
				self_.db.aDRWinParameters["Thmax"], cv2.THRESH_BINARY)[1]
				
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

	copy_image = image.copy()

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

		crop_image = crop_minAreaRect(copy_image, rect)
		#cv2.imwrite(f"data/croped_images/croped_box{i}.jpg", crop_image)
		
		box = np.int0(cv2.boxPoints(rect))
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
				self_.autoFinderWindow.auto_lines.append([box[k],box[last]])
				print(f"{self_.letter_dict[j]*mul_coef}{k+1} : {round(self_.old_determine_line([box[k],box[last]]), 2)} mm")
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

	return mask_frame, thresh














