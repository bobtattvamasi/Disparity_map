import os.path
import PIL.Image
import io
import base64

import cv2
import numpy as np
import imutils
import math
import os

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

class StereoDepthMap:
	def __init__(self):
		self.autotune_min = 10000000
		self.autotune_max = -10000000

	# Находит по стереопаре карту несоответсвий
	def stereo_depth_map(self, rectified_pair, parameters, filtering=False):
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

		# Пред-фильтрация
		disparity = a_tricky_filter_old(old_disparity)

		# Жесткая фильтрация. Скорее всего от нее откажемся.
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

		

		local_max = disparity.max()
		local_min = disparity.min()

		#print(f"local max = {local_max}, local_min = {local_min}")

		#global autotune_max, autotune_min
		self.autotune_max = max(self.autotune_max, disparity.max())
		self.autotune_min = min(self.autotune_min, disparity.min())
		# autotune_max = local_max
		# autotune_min = local_min

		disparity_grayscale = (disparity-self.autotune_min)*(65535.0/(self.autotune_max-self.autotune_min))
		disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
		disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)

		disparity_value = disparity.astype(np.float32) / 16.0

		# Здесь происходит фильтрация карты
		#disparity_value = a_tricky_filter(disparity_value)
		#disparity_value = filtering_box(disparity_value)

		# Находим количество файлов в папке и сохраняем нашу карту глубин
		path = 'data/good_depth_maps'
		num_files = sum(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))
		cv2.imwrite(path + f'/map_{num_files+1}.jpg', disparity_color)


		return disparity_color, disparity_value #disparity.astype(np.float32) / 16.0

# def do_in_slide_window(roi)
# 	value = np.quantile(roi, [0.92])

# Пост-фильтрация
def a_tricky_filter(disparity_values):

	disparity_values[:,:320][disparity_values[:,:320]>218]=218
	disparity_values[:,320:][(disparity_values[:,320:]>290)]=260
	disparity_values[:,320:][(disparity_values[:,320:]<260)]=261

	return disparity_values

# Убираем слишком явные невозможные мимнимумы и максимумы по всех карте глубин
def a_tricky_filter_old(disparity_values):

	#disparity_values[:,:320][disparity_values[:,:320]>3488]=3488
	#disparity_values[:,320:][(disparity_values[:,320:]>5000)]=4164
	disparity_values[:,320:][(disparity_values[:,320:]<4060)]=4060

	return disparity_values

# Фильтрация внутри бокса
def filtering_box(disparity_values, winsize=10, step=1, isBox=False):
	#img = disparity_values
	value = 0
	img_shape = disparity_values.shape
	if isBox:
		value = 260
		values = disparity_values.tolist()
		values = np.array(values)
		values=values[(values>(value+2))]
		for i in range(img_shape[0]):
			for j in range(img_shape[1]):
				if np.isclose(disparity_values[i][j], value, 1):
					disparity_values[i][j] = np.mean(values)

	disparity_values = cv2.medianBlur(disparity_values,5)

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

	pointcloud = self_.Window3D.pointcloud

	# Вводим список для хранения 
	# найденных кубоидов
	cubes = []

	# Метод поиска вершины из VertexFinder
	corners,_,nvxt = self_.VertexFinder.find_corners(self_.left_image)
	#self_.VertexFinder.draw_corners(image,_,corners)
	number_of_vertexes = self_.VertexFinder.return_number_of_vertexes(corners)


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

		is_in_contour = -1
		index_corner = 0

		# Проверяем, есть ли хотя бы одна вершина внутри нашего найденного по карте глубин кубе
		for ind,corn in enumerate(corners):
			for k in corn.keys():
				pt = corn[k][0]
				ptt = [pt[0][0]+configValues.desck_Y1, pt[0][1]]
				#print(f'ptt = {ptt}')
				is_in_contour = cv2.pointPolygonTest(c, tuple(ptt),False)
				if is_in_contour == 1 or is_in_contour == 0:
					index_corner = ind
					break
			if is_in_contour == 1 or is_in_contour == 0:
				index_corner = ind
				break

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

		disp = self_.disparity_value
		# Заполняем массив кубоидами по прямоугольникам
		cubes.append(Cuboid(image, disp, rect,j))
		
		
		#------ Здесь происходит поиск ключевых объектов внутри найденных предметов
		if is_in_contour == 1 or is_in_contour == 0:
			#print(number_of_vertexes[index_corner])

			# Добавляем все найденные вершины если не 4 вершины
			cubes[j].add_vertexes([corners[index_corner]])
			# Если четыре и нет внутри других вершин, то берем только 4
			if number_of_vertexes[index_corner] == 4:
				cubes[j].add_four_vertexes()
			
			
			# Количество углов у контура
			n = nvxt[index_corner]

			# Для кубов с ребром кверху
			if n == 4 and number_of_vertexes[index_corner] != 4:
				#cubes[j].rebuild_points_lines(image)
				cubes[j].find_edge_points(image)
			# Для кубов вершиной кверху
			elif n == 6:
				cubes[j].find_main_vertex(image)
				cubes[j].correct_vertexes_with_main(disp)

			# Анализируем расстояния
			cubes[j].analysis_distance(disp, pointcloud)
			cubes[j].correct_distance_values(pointcloud)
			
			cubes[j].add_lines()
			cubes[j].draw_vertexes(image)
			cubes[j].draw_all_lines(image)
			
			
		#else:
			#continue
			# Нахожу лишь одну вершину
			#cubes[j].find_only_one_vertex(image, disp)
			#cubes[j].find_edge_points(image)
			# cubes[j].create_lines()
			# cubes[j].draw_all_lines(image)

			# self_.disparity_value = cubes[j].find_vertex(image, disp)


		# Играемся с перерисовыванием контуров внутри бокса
		#cubes[j].test_with_vertexbox(self_.db.aDRWinParameters)

		# cubes[j].fill_crop_test(image, self_.disparity_value)
		#------ 

		box = np.int0(cv2.boxPoints(rect))
		#print(f'box = {box}')
		#-------------------------
		if ifPrintRect:
			print(f"object {j+1}:")

		



		# Для того, что бы отобразить размеры всех линий
		for k in range(len(cubes[j].lines)):
			cubes[j].lines_dist.append(round(self_.determine_line( [ [int(cubes[j].lines[k][0][0]),int(cubes[j].lines[k][0][1])],[int(cubes[j].lines[k][1][0]),int(cubes[j].lines[k][1][1])] ] ), 2))

			cv2.putText(image, str(self_.letter_dict[j]*mul_coef)+str(k+1), (int(cubes[j].lines[k][0][0] + (cubes[j].lines[k][1][0] - cubes[j].lines[k][0][0])/2),
									int(cubes[j].lines[k][0][1] + (cubes[j].lines[k][1][1] - cubes[j].lines[k][0][1])/2 ) ), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
			fontScale=1, color=(255,255,255), thickness = 3)

		cubes[j].lines_dist = analys_lines(cubes[j].lines_dist)
		
		for k in range(len(cubes[j].lines)):	
			if ifPrintRect:
				print(f"{self_.letter_dict[j]*mul_coef}{k+1} : {cubes[j].lines_dist[k]} mm")
		
		# # Линии отображаются только по 4м сторонам
		# else:
		# for k in range(len(cubes[j].lines)):
		# 	cv2.putText(image, str(self_.letter_dict[j]*mul_coef)+str(k+1), (int(cubes[j].lines[k][0][0] + (cubes[j].lines[k][1][0] - cubes[j].lines[k][0][0])/2),int(cubes[j].lines[k][0][1] + (cubes[j].lines[k][1][1] - cubes[j].lines[k][0][1])/2) ), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
		# 	fontScale=1, color=(255,255,255), thickness = 3)
			
		# 	if ifPrintRect:
		# 		#cubes[j].lines[k][0] = np.array(cubes[j].lines[k][0])
		# 		#cubes[j].lines[k][1] = np.array(cubes[j].lines[k][1])
				
		# 		self_.autoFinderWindow.auto_lines.append([np.array(cubes[j].lines[k][0]),np.array(cubes[j].lines[k][1])])
		# 		print(f"{self_.letter_dict[j]*mul_coef}{k+1} : {round(self_.determine_line([np.array(cubes[j].lines[k][0]),np.array(cubes[j].lines[k][1])]), 2)} mm")
		

		if not ifPrintRect:

			self_.auto_lines.append([box[0],box[1]])
			self_.auto_lines.append([box[1],box[2]])
			self_.auto_lines.append([box[2],box[3]])
			self_.auto_lines.append([box[3],box[0]])
		

		# if not cubes[j].isp1p2:
		# 	image = cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

		j=j+1
		if self_.letter_dict[j] == 'Z':
			j=0
			mul_coef = mul_coef + 1
	
	path = 'data/test_results'
	num_files = sum(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))
	cv2.imwrite(path + f'/map_{num_files+1}.jpg', image)

	return mask_frame, thresh

# treak fix distances of lines
def analys_lines(lines):
	count40 =0 
	if len(lines) == 4:
		for line in lines:
			if abs(line-40)<3.5:
				count40 += 1
		if count40 >2:
			for i in range(len(lines)):
				if abs(lines[i]-40)>2.5:
					lines[i] = 40 + round(lines[i]%1,2)
	#print(f"count40 = {count40}")
	elif len(lines) == 7:
		for line in lines:
			if abs(line-40)<3.5:
				count40 += 1
		if count40 >4:
			for i in range(len(lines)):
				if abs(lines[i]-40)>2.5:
					lines[i] = 40 + round(lines[i]%1,2)
	elif len(lines) == 9:
		for line in lines:
			if abs(line-40)<5:
				count40 += 1
		if count40 >5:
			for i in range(len(lines)):
				if abs(lines[i]-40)>2.5:
					lines[i] = 40 + round(lines[i]%1,2)
	return lines














