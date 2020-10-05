import os.path
import PIL.Image
import io
import base64

import cv2
import numpy as np
import imutils
import math

from config.config import configValues
from Objects.Cuboid import Cuboid


# Функция калибрует два изображения, разделяет их и эти два возвращает
def calibrate_two_images(imageToDisp, ifCamPi = True, calibratefolder = './data'):
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
		npzfile = np.load(calibratefolder + '/calibration_data/{}p/stereo_camera_calibration.npz'.format(image_height))
	except:
		print("Camera calibration data not found in cache, file " & calibratefolder + '/calibration_data/{}p/stereo_camera_calibration.npz'.format(image_height))
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
	
	vis=np.concatenate((imgL, imgR), axis=1)
	#cv2.imwrite('data/remapedFrame/remapedFrame.jpg', vis)
	return imgL, imgR

def resize_rectified_pair(rectified_pair):
	imgRs = cv2.resize (rectified_pair[1], dsize=(640, 362), interpolation = cv2.INTER_CUBIC)
	imgLs = cv2.resize (rectified_pair[0], dsize=(640, 362), interpolation = cv2.INTER_CUBIC)
	return imgLs, imgRs

autotune_min = 10000000
autotune_max = -10000000

# Находит по стереопаре карту несоответсвий
def stereo_depth_map(rectified_pair, parameters, filtering=False):
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
	old_disparity = disparity
	disparity = a_tricky_filter_old(old_disparity)

	if filtering:
		wls_filter = cv2.ximgproc.createDisparityWLSFilter(sbm)
		right_matcher = cv2.ximgproc.createRightMatcher(sbm)
		right_disp = right_matcher.compute(dmRight, dmLeft)
		
		wls_filter.setLambda(parameters['lambda'])	
		wls_filter.setSigmaColor(parameters['sigmaColor'])
		wls_filter.setDepthDiscontinuityRadius(parameters['Radius'])
		wls_filter.setLRCthresh(parameters['LRCthresh'])

		disparity = wls_filter.filter(disparity, dmLeft, disparity_map_right=right_disp )
		#disparity_value = filtering_box(disparity_value)
		

	#print(f"LRCthresh = {wls_filter.getLRCthresh()}")
	
	#disparity = wls_filter.getConfidenceMap()

	

	local_max = disparity.max()
	local_min = disparity.min()

	print(f"local max = {local_max}, local_min = {local_min}")

	global autotune_max, autotune_min
	# autotune_max = max(autotune_max, disparity.max())
	# autotune_min = min(autotune_min, disparity.min())
	autotune_max = local_max
	autotune_min = local_min

	disparity_grayscale = (disparity-autotune_min)*(65535.0/(autotune_max-autotune_min))
	disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
	disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)

	disparity_value = disparity.astype(np.float32) / 16.0

	disparity_value = a_tricky_filter(disparity_value)

	disparity_value = filtering_box(disparity_value)


	return disparity_color, disparity_value #disparity.astype(np.float32) / 16.0

# def do_in_slide_window(roi)
# 	value = np.quantile(roi, [0.92])

def a_tricky_filter(disparity_values):
	# print(f"disparity_max : {np.argmax(disparity_values[159:])}")
	# print(f"disparity : {disparity_values[159][10]}")

	# print(f"shape = {disparity_values.shape}")

	disparity_values[:,:320][disparity_values[:,:320]>218]=218
	disparity_values[:,320:][(disparity_values[:,320:]>290)]=260
	disparity_values[:,320:][(disparity_values[:,320:]<260)]=261

	return disparity_values

def a_tricky_filter_old(disparity_values):
	# print(f"disparity_max : {np.argmax(disparity_values[159:])}")
	# print(f"disparity : {disparity_values[159][10]}")

	# print(f"shape = {disparity_values.shape}")

	disparity_values[:,:320][disparity_values[:,:320]>3488]=3488
	disparity_values[:,320:][(disparity_values[:,320:]>5000)]=4164
	disparity_values[:,320:][(disparity_values[:,320:]<4161)]=4161

	return disparity_values


def filtering_box(disparity_values, winsize=10, step=1, isBox=False):
	#img = disparity_values
	value = 0
	img_shape = disparity_values.shape
	if isBox:
		value = 260
		#value = np.quantile(disparity_values, [0.92])
		values = disparity_values.tolist()
		values = np.array(values)
		values=values[(values>(value+2))]
		# ~ print(f"values = {values}")
		# ~ print(f"np.mean(values) = {np.mean(values)}")
		for i in range(img_shape[0]):
			for j in range(img_shape[1]):
				if np.isclose(disparity_values[i][j], value, 1):
					disparity_values[i][j] = np.mean(values)

	
	#print(f"width = {width}, height = {height}")
	# for i in range(0, int(img_shape[0]-winsize),step):
	# 	ii = i + winsize//2
	# 	for j in range(0, int(img_shape[1]-winsize),step):
	# 		jj = j+winsize//2
			
	# 		if not isBox:
	# 			value = np.quantile(disparity_values[ii-(winsize//2-1):ii+(winsize//2-1), jj-(winsize//2-1):jj+(winsize//2-1)], [0.92])
	# 		#print(f"i = {i}, j = {j}")
	# 		#print(f"value = {value}")
	# 		if abs(disparity_values[i][j] - value)>100:
	# 			disparity_values[i][j] = value

	#kernel = np.ones((5,5),np.float32)
	#disparity_values = cv2.filter2D(disparity_values,14,kernel)
	disparity_values = cv2.medianBlur(disparity_values,5)


	# img = disparity_values
	# img_shape = img.shape
	# size = 20
	# shape = (img.shape[0] - size + 1, img.shape[1] - size + 1, size, size)
	# strides = 2 * img.strides
	# patches = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
	# patches = patches.reshape(-1, size, size)

	# output_img = np.array([do_in_slide_window(roi) for roi in patches])
	# output_img.reshape(img_size)





	return disparity_values

		
	
			


# Вырезает найденный прямоугольный 
# кусок c изображения под углом.
def crop_minAreaRect(img, rect, disp=False, image=None, j=None):


	box = cv2.boxPoints(rect)
	box = np.int0(box)

	width = int(rect[1][0])
	height = int(rect[1][1])

	src_pts = box.astype("float32")

	dst_pts = np.array([[0, height-1],
						[0, 0],
						[width-1, 0],
						[width-1, height-1]], dtype="float32")

	# the perspective transformation matrix
	M = cv2.getPerspectiveTransform(src_pts, dst_pts)

	# directly warp the rotated rectangle to get the straightened rectangle
	warped = cv2.warpPerspective(img, M, (width, height))
	#cv2.imwrite(f"data/croped_images/croped_image{j+1}.jpg", warped)

	return warped

# Вырезает найденный прямоугольный 
# кусок c изображения под углом.
def return_box_picture(img, rect, box):
	width = int(rect[1][0])
	height = int(rect[1][1])

	src_pts = box.astype("float32")

	dst_pts = np.array([[0, height-1],
						[0, 0],
						[width-1, 0],
						[width-1, height-1]], dtype="float32")
	M = cv2.getPerspectiveTransform(src_pts, dst_pts)
	inv_M = cv2.getPerspectiveTransform(dst_pts, src_pts)

	pts = np.int0(cv2.transform(np.array([box]), M))[0]    
	pts[pts < 0] = 0

	rows,cols = img.shape[0], img.shape[1]
	img_rot = cv2.warpPerspective(img, M, (cols,rows))
	disp_crop = img_rot[pts[1][1]:pts[0][1], 
					   pts[1][0]:pts[2][0]]
	return disp_crop

# Автоопределение прямоугольных обектов на изображении
def autoFindRect(image, hsv_frame, self_, ifPrintRect=False):

	# Устанавливается диапазон значений для цветового фильтра 
	hsv_min = np.array((self_.db.aDRWinParameters["lowH"], self_.db.aDRWinParameters["lowS"], self_.db.aDRWinParameters["lowV"]), np.uint8)
	hsv_max = np.array((self_.db.aDRWinParameters["highH"], self_.db.aDRWinParameters["highS"], self_.db.aDRWinParameters["highV"]), np.uint8)

	# Делаем кадр серым
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	height, width = gray.shape

	# Накладываем на кадр цветовой фильтр в заданном диапазоне (hsv_min, hsv_max)
	# (правое верхнее изображение в окне настройки автонахождения кубов)
	mask_frame = cv2.inRange(hsv_frame, hsv_min, hsv_max)

	# Смазываем изображение немного
	blurred = cv2.GaussianBlur(mask_frame, (5, 5), 0)

	# Находятся пороговые значения
	thresh = cv2.threshold(blurred, self_.db.aDRWinParameters["Thmin"], 
				self_.db.aDRWinParameters["Thmax"], cv2.THRESH_BINARY)[1]

	# Tricky-Фильтр для того, что бы не искать и не детектированить ничего у столба сцены 
	thresh[:,1200:1280] = 255
	
	# Снова сглаживаем, но используем теперь срединное сглаживание
	image_blur = cv2.medianBlur(thresh, 25)
	# Снова находятся пороговые значения и меняются цветами черное с белым
	# (Правое нижнее изображение в окне настройки автонахождения кубов)
	_, thresh = cv2.threshold(image_blur, 240,255, cv2.THRESH_BINARY_INV)

	# Находим все контуры на получившемся изображении
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
							cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	
	# Номер найденного объекта
	j=0

	# Если =1, то пишется одна буква(например A),
	# Если =2, то AA
	# И т.д.
	mul_coef = 1

	# Инициализация цетра масс 
	cX = 0
	cY = 0

	# Обнуляем все автонайденные линии,
	# потому что, мы начинаем их сначала искать
	self_.auto_lines = []

	# Вводим список для хранения 
	# найденных кубоидов
	cubes = []

	for i,c in enumerate(cnts):

		# Находится площадь контура 
		contour_area = cv2.contourArea(c)
		# Отсеиваем если слишком маленькие площади
		if contour_area < 150:
			continue
		# Находим периметр контура и опять же отсеиваем слишком маленькие	
		peri = cv2.arcLength(c, True)
		if peri < 300:
			continue
		# 337-24-82

		# Считаются центры масс контуров, затем детектируется имя формы, используя только контор
		M = cv2.moments(c)
		if M["m00"] != 0:
			cX = int((M["m10"] / M["m00"]))
			cY = int((M["m01"] / M["m00"]))

		# Фильтр на игнор правой части изображения
		if cX > 1200:
			continue

		# multiply the contour (x, y)-coordinates by the resize ratio,
		# then draw the contours and the name of the shape on the image
		c = c.astype("float")
		c = c.astype("int")

		# Аппроксимируем контуры в линии
		c = cv2.approxPolyDP(c, 0.020 * peri, True)

		# Создаем прямоугольник под углом через контур
		rect = cv2.minAreaRect(c)


		#crop_image = crop_minAreaRect(copy_image, rect, j=j)

		disp = self_.disparity_value
		#if j == 1:
		
		#cv2.imwrite("point_disp.jpg", copy_image)
		
		# Заполняем массив кубоидами по прямоугольникам
		cubes.append(Cuboid(image, disp, rect,j))
		
		#------ Здесь происходит поиск ключевых объектов внутри найденных предметов

		# Нахожу лишь одну вершину
		cubes[j].find_only_one_vertex(image)

		self_.disparity_value = cubes[j].find_vertex(image, disp)

		cubes[j].fill_crop_test(image, self_.disparity_value)
		#------ 

		box = np.int0(cv2.boxPoints(rect))
		#-------------------------
		if ifPrintRect:
			print(f"object {j+1}:")
		# for printing all lines of object
		if not cubes[j].isp1p2:
			for k in range(4):
				last = k+1
				if last ==4:
					last = 0			
				cv2.putText(image, str(self_.letter_dict[j]*mul_coef)+str(k+1), (int(box[k][0] + (box[last][0] - box[k][0])/2),int(box[k][1] + (box[last][1] - box[k][1])/2) ), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
				fontScale=1, color=(255,255,255), thickness = 3)
				
				if ifPrintRect:
					#print(f"box type = {type(box[k])}")
					self_.autoFinderWindow.auto_lines.append([box[k],box[last]])
					print(f"{self_.letter_dict[j]*mul_coef}{k+1} : {round(self_.old_determine_line([box[k],box[last]]), 2)} mm")
		else:
			for k in range(len(cubes[j].lines)):
				cv2.putText(image, str(self_.letter_dict[j]*mul_coef)+str(k+1), (int(cubes[j].lines[k][0][0] + (cubes[j].lines[k][1][0] - cubes[j].lines[k][0][0])/2),int(cubes[j].lines[k][0][1] + (cubes[j].lines[k][1][1] - cubes[j].lines[k][0][1])/2) ), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
				fontScale=1, color=(255,255,255), thickness = 3)
				
				if ifPrintRect:
					#print(f"list_of_lines[k][0]= {list_of_lines[k][0]}")
					#print(f"list_of_lines[k][1] = {list_of_lines[k][1]}")
					cubes[j].lines[k][0] = np.array(cubes[j].lines[k][0])
					cubes[j].lines[k][1] = np.array(cubes[j].lines[k][1])
					#print(f"list_of_lines type = {type(list_of_lines[k][0])}")
					
					# ~ if k == len(list_of_lines)-1:
							# ~ disparity_line_point			
					
					self_.autoFinderWindow.auto_lines.append([np.array(cubes[j].lines[k][0]),np.array(cubes[j].lines[k][1])])
					print(f"{self_.letter_dict[j]*mul_coef}{k+1} : {round(self_.smart_determine_line([np.array(cubes[j].lines[k][0]),np.array(cubes[j].lines[k][1])]), 2)} mm")
					#self_.smart_determine_line([np.array(list_of_lines[k][0]),np.array(list_of_lines[k][1])])
		

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
		if not cubes[j].isp1p2:
			image = cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
		#cv2.imwrite("rottate_rect.jpg", image)

		j=j+1
		if self_.letter_dict[j] == 'Z':
			j=0
			mul_coef = mul_coef + 1

	return mask_frame, thresh














