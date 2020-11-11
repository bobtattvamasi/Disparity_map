from tools import ImageProccessHelper
from Objects.BaseObject import BaseObject
import cv2
import numpy as np
#from tools.ConstructProjectHelper import timer
from config.config import configValues as cfv


class Cuboid(BaseObject):

	# def add_ed_box_point_clod(self, pointcloud):
	# 	pass

	# def plane_fitting(self):
	# 	pass

	def small_diff_z_in_square(self):
		z1 = self.analysis_dict[0][1]
		z2 = self.analysis_dict[1][1]
		z3 = self.analysis_dict[2][1]
		z4 = self.analysis_dict[3][1]

		minus1 = abs(z1-z2)
		minus2 = abs(z1-z3)
		minus3 = abs(z1-z4)

		if max([minus1, minus2, minus3]) < 2.1:
			return True
		else:
			return False

	def analysis_distance(self, disp_value, pointcloud):
		for i, v in enumerate(self.vertexes):
			# self.analysis_dict содержит:
			# 1. значение карты глубины в точке
			self.analysis_dict[i] = [disp_value[int(v[1])][int(v[0])], 
									pointcloud[int(v[1])][int(v[0])][2]]
		#print(f'self.analysis_dict = {self.analysis_dict}')

	def correct_distance_values(self, pointcloud):
		# Плоские квадраты(верхние грани)
		if len(self.vertexes) == 4:
			if self.small_diff_z_in_square():

				z_list = []
				for pcz in self.analysis_dict.items():
					z_list.append(pcz[1][1])
				av_z = sum(z_list) / len(z_list)
				for v in self.vertexes:
					pointcloud[int(v[1])][int(v[0])][2] = av_z

		elif len(self.vertexes) == 6:

			z_a1 = self.analysis_dict[0][1]
			z_a2 = self.analysis_dict[1][1]
			av_za = (z_a1+z_a2) / 2

			z_c1 = self.analysis_dict[2][1]
			z_c2 = self.analysis_dict[5][1]
			av_zc = (z_c1+z_c2) / 2

			z_b1 = self.analysis_dict[3][1]
			z_b2 = self.analysis_dict[4][1]
			av_zb = (z_b1+z_b2) / 2

			#if av_za < av_zb:
			pointcloud[int(self.vertexes[0][1])][int(self.vertexes[0][0])][2] = av_za
			pointcloud[int(self.vertexes[1][1])][int(self.vertexes[1][0])][2] = av_za

			pointcloud[int(self.vertexes[2][1])][int(self.vertexes[2][0])][2] = av_zc
			pointcloud[int(self.vertexes[5][1])][int(self.vertexes[5][0])][2] = av_zc

			pointcloud[int(self.vertexes[3][1])][int(self.vertexes[3][0])][2] = av_zb
			pointcloud[int(self.vertexes[4][1])][int(self.vertexes[4][0])][2] = av_zb
			#for pcz in self.analysis_dict.items():
	
	# Оставляем
	#@timer
	def _crop_object(self, image):
		src_pts = self.box.astype("float32")

		dst_pts = np.array([[0, self.height-1],
							[0, 0],
							[self.width-1, 0],
							[self.width-1, self.height-1]], dtype="float32")

		# the perspective transformation matrix
		M = cv2.getPerspectiveTransform(src_pts, dst_pts)

		# directly warp the rotated rectangle to get the straightened rectangle
		warped = cv2.warpPerspective(image, M, (self.width, self.height))
		#cv2.imwrite(f"data/croped_images/croped_image{self.number+1}.jpg", warped)
		return warped

	# Выкидываем
	def _return_box_picture(self, img):
		width = int(self.rect[1][0])
		height = int(self.rect[1][1])

		src_pts = self.box.astype("float32")

		dst_pts = np.array([[0, height-1],
							[0, 0],
							[width-1, 0],
							[width-1, height-1]], dtype="float32")
		M = cv2.getPerspectiveTransform(src_pts, dst_pts)
		inv_M = cv2.getPerspectiveTransform(dst_pts, src_pts)

		pts = np.int0(cv2.transform(np.array([self.box]), M))[0]    
		pts[pts < 0] = 0

		rows,cols = img.shape[0], img.shape[1]
		img_rot = cv2.warpPerspective(img, M, (cols,rows))
		disp_crop = img_rot[pts[1][1]:pts[0][1], 
						   pts[1][0]:pts[2][0]]
		#cv2.imwrite(f"data/croped_images/returned_box_picture{self.number+1}.jpg", disp_crop)
		return disp_crop

	def test_with_vertexbox(self, params):
		if self.main_vertex is not None:

			# Устанавливается диапазон значений для цветового фильтра 
			hsv_min = np.array((params["lowH"], params["lowS"], params["lowV"]), np.uint8)
			hsv_max = np.array((params["highH"], params["highS"], params["highV"]), np.uint8)

			gray = cv2.cvtColor(self.Mat_image, cv2.COLOR_BGR2GRAY)

			hsv_frame = cv2.cvtColor(self.Mat_image, cv2.COLOR_BGR2HSV)
			mask_frame = cv2.inRange(hsv_frame, hsv_min, hsv_max)

			blurred = cv2.GaussianBlur(mask_frame, (5, 5), 0)

			thresh = cv2.threshold(blurred, params["Thmin"], 
					params["Thmax"], cv2.THRESH_BINARY)[1]
			
			image_blur = cv2.medianBlur(thresh, 25)
			_, thresh = cv2.threshold(image_blur, 240,255, cv2.THRESH_BINARY_INV)

			#cv2.imwrite(f'data/work_on_contour_redraw/cube_mask_{self.number}.jpg', thresh)

	def correct_vertexes_with_main(self, disp_value):
		disp0 = disp_value[self.vertexes[0][1]][self.vertexes[0][0]]
		disp1 = disp_value[self.vertexes[1][1]][self.vertexes[1][0]]

		if disp0 > disp1:
			self.main_case = 0
		else:
			self.main_case = 1

	def find_main_vertex(self, image):
		

		big_black_image = np.zeros((720,1280, 3), np.uint8)

		listOfCoordinates = self.get_disparity_values_by_last_quart()
		rotated_indexes = []
		for index in listOfCoordinates:
			index = self.rotate_xy_from_rect_coord_system(index)
			rotated_indexes.append(index)
			self.draw_point(big_black_image,index, color=(255,255,255))
			#self.draw_point(image,index, color=(255,255,255))

		big_gray = cv2.cvtColor(big_black_image, cv2.COLOR_BGR2GRAY)
		big_blur = cv2.GaussianBlur(big_gray, (3,3), 0)

		big_thresh = cv2.threshold(big_blur, 240,255, cv2.THRESH_BINARY)[1]

		big_cnts = cv2.findContours(big_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		big_cnts = big_cnts[0] if len(big_cnts) == 2 else big_cnts[1]
		cv2.drawContours(big_black_image, big_cnts, -1, (0, 255, 0), 2)
		#self.draw_text_point(big_black_image, self.vertexes[6], " 6", textcolor=(0,0,255))
		#self.draw_text_point(big_black_image, self.vertexes[7], " 7", textcolor=(0,0,255))
		#cv2.imwrite(f'data/big_black_images/cube_mask_{self.number}.jpg', big_black_image)
		big_c = max(big_cnts, key=cv2.contourArea)

		new_vertexes = self.vertexes[:6]

		dist6 = cv2.pointPolygonTest(big_c, tuple(self.vertexes[6]),True)
		dist7 = cv2.pointPolygonTest(big_c, tuple(self.vertexes[7]),True)
		if dist6 > dist7:
			#print(6)
			new_vertexes.append(self.vertexes[6])
		else:
			#print(7)
			new_vertexes.append(self.vertexes[7])

		self.vertexes = new_vertexes



	# Функция где мы находим линию ребра
	def find_edge_points(self, image):
		if self.main_vertex is None:

			# Найти на картинке бокса особенные точки
			#height, width = self.Mat_image.shape[:2]
			#black_image = np.zeros((height,width, 3), np.uint8)
			big_black_image = np.zeros((720,1280, 3), np.uint8)
			#black_image[:] = (255,0,0)
			# Нарисовать эти точки на черном изображении того же размера

			# Работать с этими точками...
			listOfCoordinates = self.get_disparity_values_by_last_quart()
			rotated_indexes = []

			for index in listOfCoordinates:
				#self.draw_point(black_image, index, color=(255,255,255))
				index = self.rotate_xy_from_rect_coord_system(index)
				rotated_indexes.append(index)
				self.draw_point(big_black_image,index, color=(255,255,255))
				#self.draw_point(image,index, color=(255,255,255))

			big_gray = cv2.cvtColor(big_black_image, cv2.COLOR_BGR2GRAY)
			big_blur = cv2.GaussianBlur(big_gray, (3,3), 0)

			big_thresh = cv2.threshold(big_blur, 240,255, cv2.THRESH_BINARY)[1]

			big_cnts = cv2.findContours(big_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			big_cnts = big_cnts[0] if len(big_cnts) == 2 else big_cnts[1]
			cv2.drawContours(big_black_image, big_cnts, -1, (0, 255, 0), 2)
			#cv2.imwrite(f'data/big_black_images/cube_mask_{self.number}.jpg', big_black_image)
			big_c = max(big_cnts, key=cv2.contourArea)


			#print(self.vertexes)
			new_vertexes = self.vertexes[:2]
			second_part_of_verxes = self.vertexes[2:4]
			#print(second_part_of_verxes)
			ii = []
			for i,pt in enumerate(self.vertexes[4:]):
				aa = cv2.pointPolygonTest(big_c, tuple(pt),False)
				if aa == 0 or aa == 1:
					#print(f"i  = {i+4}, pointPolygonTest = {aa}")
					ii.append(i+4)
			edge45 = 0
			edge67 = 0
			if 4 in ii:
				edge45 += 1
			if 5 in ii:
				edge45 += 1
			if 6 in ii:
				edge67 += 1
			if 7 in ii:
				edge67 += 1

			#print (f"edge45 = {edge45}, edge67 = {edge67}")
			if edge45 > edge67:
				new_vertexes.append(self.vertexes[5])

				new_vertexes.append(second_part_of_verxes[0])
				new_vertexes.append(second_part_of_verxes[1])

				new_vertexes.append(self.vertexes[4])
				#print("edge_4,5")
			elif edge45 < edge67:
				new_vertexes.append(self.vertexes[7])

				new_vertexes.append(second_part_of_verxes[0])
				new_vertexes.append(second_part_of_verxes[1])

				new_vertexes.append(self.vertexes[6])
				#print("edge_6,7")
			else:
				new_vertexes.append(second_part_of_verxes[0])
				new_vertexes.append(second_part_of_verxes[1])

			self.vertexes = new_vertexes

	
	def rebuild_points_lines(self, image):
		pass
		# new_vertexes = []
		# self.edge_case = []
		# p1 = self.rotate_xy_from_rect_coord_system(self.edge_points[0])
		# p2 = self.rotate_xy_from_rect_coord_system(self.edge_points[1])
		# for i in range(len(self.vertexes)):
		# 	k=i+1
		# 	if k == 4:
		# 		k=0
		# 	print(f'i = {i}')
		# 	print(f'k = {k}')
		# 	point = self.line_intersection([self.vertexes[i],self.vertexes[k]], [p1,p2])
		# 	#print(f'point = {point}')
		# 	if point:
		# 		#print(f'cases = {i}')
		# 		new_vertexes.append([point[0],point[1]])
		# 		self.edge_case.append(i)
		
		# if len(new_vertexes) == 2:
		# 	#print(f'casecase = {self.edge_case[0]}')
		# 	k=0
		# 	if self.edge_case[0] == 1:
		# 		copy_vertexes = self.vertexes
		# 		self.vertexes = []
		# 		for i in range(len(copy_vertexes)):
		# 			self.vertexes.append(copy_vertexes[i])
		# 			if i%2 == 1:
		# 				self.vertexes.append(new_vertexes[k])
		# 				k=k+1
					
		# 	elif self.edge_case[0] == 0:
		# 		copy_vertexes = self.vertexes
		# 		self.vertexes = []
		# 		for i in range(len(copy_vertexes)):
		# 			self.vertexes.append(copy_vertexes[i])
		# 			if i%2 == 0:
		# 				self.vertexes.append(new_vertexes[k])
		# 				k=k+1
		#self.draw_circle(image, self.vertexes[4])
		# self.draw_text_point(image, self.vertexes[4], " 4", textcolor=(0,0,255))
		# self.draw_text_point(image, self.vertexes[5], " 5", textcolor=(0,0,255))
		# self.draw_text_point(image, self.vertexes[6], " 6", textcolor=(0,0,255))
		# self.draw_text_point(image, self.vertexes[7], " 7", textcolor=(0,0,255))
				 
		


	def __init__(self, image, disparity_values, rect, number):
		# rotated rect
		self.rect = rect
		self.box = np.int0(cv2.boxPoints(rect))
		
		self.width = int(rect[1][0])
		self.height = int(rect[1][1])
		
		self.lines = []
		self.lines_dist = []

		self.number = number
		
		self.Mat_disparity_crop = self._crop_object(disparity_values)
		self.Mat_image = self._crop_object(image)
		self.Mat_picture = self._crop_object(image)

		
		
		#Все найденные вершины будут храниться здесь
		self.vertexes = []
		self.main_vertex = None
		# Угол между нижней вертикалью и нижним ребром найденного кубоида 
		self.rectangle_angle = self._angle_between_two_lines((self.box[1], self.box[2]), (self.box[1],(self.box[1][0]+10, self.box[1][1])))
		
		# All about inner edge line
		self.edge_points = [None,None]
		self.disparity_line_points = []
		self.isp1p2 = False
		
		self.edge_case = []
		self.main_case = 0

		# About point clouds
		self.pointcloud = None
		self.analysis_dict = {}

	def add_vertexes(self, corners):
		for i in corners:
			for k in i.keys():
				pt = i[k][0]
				self.vertexes.append([pt[0][0]+cfv.desck_Y1, pt[0][1]])

	def add_four_vertexes(self):
		self.vertexes = self.vertexes[:4]

	def add_lines(self):
		if len(self.vertexes) == 4:
			for i in range(len(self.vertexes)):
				k=i+1
				if k==4:
					k=0
				self.lines.append([self.vertexes[i], self.vertexes[k]])

		elif len(self.vertexes) == 7:
			for i in range(7):
				k=i+1

				if k==7 and self.main_case == 1:
					k=1
				elif k==7 and self.main_case == 0:
					k=0

				if k==6 and self.main_case == 0:
					self.lines.append([self.vertexes[4], self.vertexes[k]])
				elif k==6 and self.main_case == 1:
					self.lines.append([self.vertexes[i], self.vertexes[k]])
				else:
					self.lines.append([self.vertexes[i], self.vertexes[k]])

			self.lines.append([self.vertexes[0], self.vertexes[5]])

			if self.main_case == 1:
				self.lines.append([self.vertexes[3], self.vertexes[6]])
			else:
				self.lines.append([self.vertexes[2], self.vertexes[6]])
		
		elif len(self.vertexes) == 6:
			for i in range(6):
				k=i+1
				if k==6:
					k=0
				self.lines.append([self.vertexes[i], self.vertexes[k]])
			self.lines.append([self.vertexes[2], self.vertexes[5]])
				
			# if self.edge_case[0] == 1:
			# 	self.lines.append([self.vertexes[2], self.vertexes[5]])	
			# elif self.edge_case[0] == 0:
			# 	self.lines.append([self.vertexes[1], self.vertexes[4]])
	
	def draw_vertexes(self,image):
		for i,v in enumerate(self.vertexes):
			cv2.circle(image, (int(v[0]), int(v[1])), 6, (255, 0, 255), 3)
				

		
	@property
	def average_disparity_value(self):
		return np.average(self.Mat_disparity_crop)
	@property
	def max_disparity_value(self):
		return np.argmax(self.Mat_disparity_crop)
	@property
	def min_disparity_value(self):
		return np.argmin(self.Mat_disparity_crop)

	def get_quantiles(self,quartos=[0.25,0.5,0.8,0.88]):
		return np.quantile(self.Mat_disparity_crop, quartos)
	
	def get_disparity_values_by_last_quart(self):
		quart = self.get_quantiles()
		indexes = np.where(self.Mat_disparity_crop >= quart[-1])
		listOfIndexes = list(zip(indexes[1], indexes[0]))
		return listOfIndexes

	def create_lines(self):
		if self.edge_points == [None,None]:
			for j in range(len(self.box)):
				last = j+1
				if last ==4:
					last = 0
				self.lines.append([self.box[j],self.box[last]])
		else:
			# for j in range(len(box)):
			# 	last = j+1
			# 	if last ==4:
			# 		last = 0
			# 	self.lines.append([box[j],box[last]])

			edge_p0 = self.rotate_xy_from_rect_coord_system(self.edge_points[0])
			edge_p1 = self.rotate_xy_from_rect_coord_system(self.edge_points[1])

			self.lines.append([self.box[0], self.box[1]])
			self.lines.append([self.box[1], edge_p0])
			self.lines.append([edge_p0, self.box[2]])
			self.lines.append([self.box[2], self.box[3]])
			self.lines.append([self.box[3], edge_p1])
			self.lines.append([edge_p1, self.box[0]])
			self.lines.append([edge_p0, edge_p1])

	def draw_all_lines(self,image):
		#print(f"LINES : {self.lines}")
		for i in range(len(self.lines)):
			self.draw_line(image, self.lines[i][0],self.lines[i][1])

			# cv2.putText(image, str(self_.letter_dict[j]*mul_coef)+str(k+1), (int(box[k][0] + (box[last][0] - box[k][0])/2),int(box[k][1] + (box[last][1] - box[k][1])/2) ), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
			# 	fontScale=1, color=(255,255,255), thickness = 3)

			# print(f"{self_.letter_dict[j]*mul_coef}{k+1} : {round(self_.determine_line([box[k],box[last]]), 2)} mm")

	# Находит максимальные точки внутри кубоида
	def find_extremum_points(self):
		# Применяем medianBlur-фильтр, сглаживая все пиксели внутри
		self.Mat_disparity_crop = ImageProccessHelper.filtering_box(self.Mat_disparity_crop, isBox=False)

		# Возвращает индексы пикселей с наивысшими значениями, которые входят 8 процентов пикселей
		# и создаем из этого массив индексов
		listOfCoordinates = self.get_disparity_values_by_last_quart()


		# Создаем массив значений диспарити карты по всем точкам линии
		# self.disparity_line_points = []
		# line_point = self._get_all_points_of_line(self.edge_points[0],self.edge_points[1])
		# self.disparity_line_points = np.array(line_point)		

		return listOfCoordinates

	# Фильтр для бокса внутри 
	def fill_crop_test(self, image, disp):
		# Находим только пиксели относящиеся к кубоиду
		only_cube_values = []
		for pix in self.Mat_disparity_crop:
			for t in range(len(pix)):
				if pix[t] >262:
					only_cube_values.append(pix[t])
		# Находим среднее по ним
		average_only_cube_value = np.average(only_cube_values)
		# Заменяем все пиксели осносящиеся к столу на найденное среднее
		self.Mat_disparity_crop[self.Mat_disparity_crop<262] = average_only_cube_value

		for i in range(self.width-1):
			for j in range(self.height-1):
				point = self.rotate_xy_from_rect_coord_system([i,j])
				point = self.boundary_condition(point)
				#self.draw_circle(image, point)
				#print(f"point = {point}")
				# if int(point[0]) == 720:
				# 	point[0] = 719
				disp[int(point[1])][int(point[0])] = self.Mat_disparity_crop[j][i]

		#print(f"line1={_self.determine_line([self.Mat_disparity_crop()])}")

	# Нахожу лишь одну вершину в кубоиде
	def find_only_one_vertex(self, image, disp):
		# создаю массив средних значений внутри границ найденного кубоида
		b = np.array([[np.mean(self.Mat_disparity_crop[y-10:y+10, x-10:x+10]) for y in range(10, int(self.height)-10)] for x in range(10, int(self.width)-10)])
		# Находим координаты x и y максимальной точки из найденного массива
		maxcenterx = np.unravel_index(b.argmax(), b.shape)[0]+10
		maxcentery = np.unravel_index(b.argmax(), b.shape)[1]+10

		point = [maxcenterx,maxcentery]
		# Масштабируем координаты на все изображение
		point = self.rotate_xy_from_rect_coord_system(point)
		
		# Присваиваем в main_vertex
		self.main_vertex = point

		# Проверка - действительно ли наша вершина - вершина 
		flag_ = self.is_main_vertex(disp)
		if disp[int(self.main_vertex[1])][int(self.main_vertex[0])] > 268 and flag_:
			# Рисуем эту вершину и подписываем 'Vertex'
			self.draw_text_point(image,self.main_vertex, 'vertex',textcolor=(0,0,255))
		else:
			self.main_vertex = None

	# Находятся экстремумы, вершины, ребра.
	def find_vertex(self, image, disp):
		pass

		# Находим индексы экстремальных точек
		# ~ indexes = self.find_extremum_points()

		# ~ # Если вершины нет, то находим внутренее ребро и отрисовываем его
		# ~ if self.main_vertex == None:
			# ~ self.find_edge_points(image)
			# ~ p1,p2 = self.edge_points
			# ~ print(f"p1,p2 = {p1}, {p2}")

			# ~ #if not is_line_out_of_box(p1,p2, box):
			# ~ if (p1 != None) and (p2 != None):
				# ~ p1 = self.rotate_xy_from_rect_coord_system(p1)
				# ~ p2 = self.rotate_xy_from_rect_coord_system(p2)

				# ~ self.draw_line(image, p1, p2)
		
		# ~ #disp = self.rebuld_disparity_map_by_rect(disp)

		# ~ disp_values_in_extremum = []
		# ~ for index in indexes:
			# ~ index = self.rotate_xy_from_rect_coord_system(index)

			# ~ # Рисуются точки экстремума на карте
			# ~ #self.draw_circle(image, index)
			# ~ disp_values_in_extremum.append(disp[int(index[1]),int(index[0])])
		
		#self.isp1p2 = False
		#list_of_lines = []
		
		
		
			#list_of_lines = self.combine_line_in_box(image, p1, p2)
			#self.isp1p2 = True

			# Переделать
			#self.isp1p2 = False

		#newpts = self.rotate_xy_from_rect_coord_system(vertex_point)
		
		# width = int(self.rect[1][0])
		# height = int(self.rect[1][1])
		# flag = True

		#draw_test_rectangle(image, rect, box)
		#coef = 0.1
		# ~ for i in range(0, int(width)):
			# ~ for j in range(0, int(height)):
				# ~ # if (newpts[0]<(coef*width+box[1][0]) and newpts[1]<(coef*height + box[1][1])) or\
				# ~ # 	(newpts[0]>((1.0 - coef)*width+box[1][0]) and newpts[1]<((1.0 - coef)*height + box[1][1])):
				# ~ p = self.rotate_xy_from_rect_coord_system([i,j])
				# ~ if (p[0]<(coef*width+box[1][0]) and p[1]<(coef*height + box[1][1])) or\
					# ~ (p[0]>((1.0 - coef)*width+box[1][0]) and p[1]>((1.0 - coef)*height + box[1][1])):
						# ~ #flag = False
						# ~ draw_point(image, p)
		# ~ if flag:
			# ~ create_lines_vertex(image, box, newpts)
		#print(f"newpts = {newpts}")
		
		#draw_text_point(image, newpts, 'vertex')
		
		#self.lines = np.int0(list_of_lines)

		return disp

	# Граничные условия для координат линий
	@staticmethod
	def boundary_condition(A):
		if A[1]>=cfv.IMAGE_HEIGHT:
			A[1] = cfv.IMAGE_HEIGHT - 1
		if A[0] >= cfv.IMAGE_WIDTH:
			A[0]=cfv.IMAGE_WIDTH - 1
		return A
	
	# Создает две точки для прямой
	@staticmethod
	def _create_line_points_by_mb(m,b,x1,x2):
		p1 = [int(m*x1 + b),int(x1)]
		p2 = [int(m*x2 + b),int(x2)]

		return [p1, p2]

	# Аппроксимируется прямая по найденому облаку экстремумов точек
	@staticmethod
	def _best_fit(points):
		x = points[:,1]
		y = points[:,0]

		m,b = np.polyfit(x,y,1)
		return m,b

	@staticmethod
	def get_four_lines_from_box(box):
		lines = []
		for i in range(4):
			j=0 if i==3 else i+1
			lines.append((box[i], box[j]))
		return lines
		
	# combine finding line p1p2 and box together
	def combine_line_in_box(self,image, p1,p2):
		box_lines = self.get_four_lines_from_box(self.box)
		list_of_return_lines = []
		dot_line = {}
		lines_to_draw = []
		
		for line in box_lines:
			xy = self.line_intersection((p1,p2), line)
			inter = self.intersect(p1,p2, line[0], line[1])
			if inter:
				x,y = xy
				dot_line.update({(x,y):line})
			else:
				lines_to_draw.append(line)
				
		for dot in dot_line:
			p1_dist = cv2.norm(np.array(p1) - np.array(dot))
			p2_dist = cv2.norm(np.array(p2) - np.array(dot))
			b1, b2 = dot_line[dot]
			if p1_dist > p2_dist:
				self.draw_line(image, p2, b1)
				self.draw_line(image, p2, b2)
				list_of_return_lines.append([p2,b1])
				list_of_return_lines.append([p2,b2])
			else:
				self.draw_line(image, p1, b1)
				self.draw_line(image, p1, b2)
				list_of_return_lines.append([p1,b1])
				list_of_return_lines.append([p1,b2])
		
		for line in lines_to_draw:
			self.draw_line(image,line[0], line[1])
			list_of_return_lines.append(line)
		list_of_return_lines.append([list(p1),list(p2)])
		
		return list_of_return_lines

	# Тут прописываем те признаки по которым, мы определяем мы нашли вершину или нет
	def is_main_vertex(self,disp, winsize=20, value = 0.35):

		# Берем область вершины с радиусом winsize
		disp_crop = disp[int(self.main_vertex[1]-winsize//2):int(self.main_vertex[1]+winsize//2),int(self.main_vertex[0]-winsize//2):int(self.main_vertex[0]+winsize//2)]

		# Находим среднее 
		mean = np.quantile(disp_crop,[0.5])[0]
		# и максимальное значение
		ma_ =  disp_crop.max()

		# Если максимальное значение больше среднего по округе на определенную величину - то у нас вершина
		#print(abs(ma_-mean)>value)
		if abs(ma_- mean)>value:
			return True
		return False

	# Точка внутри прямоуголькика? 
	# Need to test
	@staticmethod
	def _rectContains(rect_box,pt):
		logic = rect_box[0] < pt[0] < rect_box[0]+rect_box[2] and rect_box[1] < pt[1] < rect_box[1]+rect_box[3]
		return logic

	# Рисует тестовый прямоугольник что бы убедиться,
	# что rotate_xy_from_rect_coord_system работает стабильно
	def _draw_test_rectangle(self, image, rect, box):
		width = int(rect[1][0])
		height = int(rect[1][1])
		for i in range(0, width):
			for j in range(0, height):
				x,y = self.rotate_xy_from_rect_coord_system([i,j])
				draw_point(image, [x,y])


	def rebuld_disparity_map_by_rect(self, disp):
		shape = self.Mat_disparity_crop.shape
		for i in range(shape[0]-1):
			for j in range(shape[1]-1):
				value = self.Mat_disparity_crop[i][j]
				x,y = self.rotate_xy_from_rect_coord_system([i,j])
				#draw_point(image, [x,y])
				disp[int(y)][int(x)] = value
		return disp

	# (!!!BAD!!!)Рисует на найденной вершине внутри бокса линии,
	# соединяющие ее с другими четырьмя вершинами
	@staticmethod
	def _create_lines_vertex(img, box, vertex_point):
		for point in box:
			cv2.line(img, (int(vertex_point[0]), int(vertex_point[1])), tuple(point), (0,255,0), 3)

	# Находит угол между двумя линиями
	@staticmethod
	def _angle_between_two_lines(line1, line2):
		v_a = np.array(line1[1]) - np.array(line1[0])
		v_b = np.array(line2[1]) - np.array(line2[0])
		return np.degrees(np.arccos(np.dot(v_a,v_b) / (np.linalg.norm(v_a) * np.linalg.norm(v_b))))

	# Пересчитывает значения координат в зависимости от угла и начала угла бокса
	def rotate_xy_from_rect_coord_system(self,ind):
		r = self.box[1]
		a = self.rectangle_angle
		a = np.radians(-a)
		sin = np.sin(a)
		cos = np.cos(a)
		
		newx = ind[0]*cos - ind[1]*sin + r[0]
		newy = ind[1]*cos + ind[0]*sin + r[1]

		return [newx, newy]

	# Рисует заметный кружочек с надписью
	@staticmethod
	def draw_text_point(img, point, text="TEXT1", textcolor=((255,255,255))):
		cv2.circle(img, (int(point[0]), int(point[1])), 6, (255,255,0), 3)
		cv2.circle(img, (int(point[0]), int(point[1])), 3, (0,255,255), 2)
		cv2.putText(img,text,(int(point[0]),int(point[1]+3)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1, color=textcolor, thickness = 3)

	# Рисует точку
	@staticmethod
	def draw_point(img, point, color=(0,0,255)):
		cv2.circle(img, (int(point[0]), int(point[1])), 1, color, 1)

	# Рисует маленькую окружность
	@staticmethod
	def draw_circle(img, point, color=(255,255,255)):
		cv2.circle(img, (int(point[0]), int(point[1])), 3, color, 2)
		#cv2.putText(img,'max',(int(point[0]),int(point[1]+3)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1, color=(255,255,255), thickness = 3)

	# Рисуется прямая по двум точкам
	@staticmethod
	def draw_line(image, p1, p2, color=(0,255,0)):	
		cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 5)
		# cv2.putText(image,'P1',(int(p1[0]),int(p1[1]+3)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1, color=(255,255,255), thickness = 3)
		# cv2.putText(image,'P2',(int(p2[0]),int(p2[1]+3)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1, color=(255,255,255), thickness = 3)
	
	@staticmethod
	def _ccw(A,B,C):
		return (C[1] - A[1])*(B[0]-A[0])>(B[1]-A[1])*(C[0]-A[0])
		
	def intersect(self,A,B,C,D):
		return self._ccw(A,C,D) != self._ccw(B,C,D) and self._ccw(A,B,C) != self._ccw(A,B,D)

	# Пересекаются ли две линии(on infinity)
	@staticmethod
	def line_intersection(line1, line2):
		xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
		ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

		def det(a, b):
			return a[0] * b[1] - a[1] * b[0]

		div = det(xdiff, ydiff)
		if div == 0:
		   return False

		d = (det(*line1), det(*line2))
		x = det(d, xdiff) / div
		y = det(d, ydiff) / div
		array = [[line1[0]], [line2[0]], [line1[1]], [line2[1]]]
		# contour = [np.array([[line1[0]], [line2[0]], [line1[1]], [line2[1]]], dtype=np.int32)]
		contour = np.array(array).reshape((-1,1,2)).astype(np.int32)
		is_in_contour = cv2.pointPolygonTest(contour, (x,y),False)
		if is_in_contour == -1 or is_in_contour == 0:
			return False
		return x,y

	# Пересекает ли линия entry edge c линии бокса
	@staticmethod
	def is_line_intersect_box(p1,p2,box):
		for i in range(4):
			j=0 if i==3 else i+1
			coord = self.line_intersection((p1,p2), (box[i],box[j]))
			if coord:
				# x,y = coord
				return True
		return False

	# Получить индексы всех точек прямой от точки p1 до p2
	@staticmethod
	def _get_all_points_of_line(p1,p2):
		x1, y1= int(p1[0]), int(p1[1])
		x2, y2 = int(p2[0]), int(p2[1])
		points = []
		issteep = abs(y2-y1) > abs(x2-x1)
		if issteep:
			x1, y1 = y1, x1
			x2, y2 = y2, x2
		rev = False
		if x1 > x2:
			x1, x2 = x2, x1
			y1, y2 = y2, y1
			rev = True
		deltax = x2 - x1
		deltay = abs(y2-y1)
		error = int(deltax / 2)
		y = y1
		ystep = None
		if y1 < y2:
			ystep = 1
		else:
			ystep = -1
		for x in range(x1, x2 + 1):
			if issteep:
				points.append((y, x))
			else:
				points.append((x, y))
			error -= deltay
			if error < 0:
				y += ystep
				error += deltax
		# Reverse the list if the coordinates were reversed
		if rev:
			points.reverse()
		return points
	
