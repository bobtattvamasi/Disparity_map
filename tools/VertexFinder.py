import cv2
import os
import numpy as np
from config.config import configValues

class VertexFinder:
	def __init__(self):
		self.MINDIST = 30
		self.MINANGLE = 20
		self.Y1, self.Y2, self.X1, self.X2 = configValues.desck_Y1, configValues.desck_Y2, 0, 700
		self.back_ground_image = cv2.imread(configValues.background_image)
		photo_height = configValues.PHOTO_HEIGHT
		image_width = configValues.IMAGE_WIDTH
		self.back_ground_image = self.back_ground_image[0:photo_height,0:image_width]

	def run(self, image, disp_image):
		
		answers, todraw, approx = self.find_corners(img_in)
		return self.draw_corners(disp_image, todraw, answers)

	def find_corners(self, image):
		todraw = {
		'contours'    : [],
		'approxes'    : [],
		'approxes_bad_after_fixes': [],
		'approxes_bad_before_fixes': []
	}
		image = image[self.X1:self.X2, self.Y1:self.Y2]
		edges = self.find_edges_by_bkg(image)
		edges = cv2.dilate(edges, (9, 9))
		contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		answers = []
		nvxt = []
		for i, contour in enumerate(contours):
			# x, y, w, h = cv2.boundingRect(contour)
			area = cv2.contourArea(contour)
			if 3000 < area and area < 20000:
				# print('========'*20)
				# img_in = cv2.drawContours(img_in, [contour],  -1, (0, 255, 0), thickness=3)
				contour_convex = cv2.convexHull(contour)
				epsilon = 0.03*cv2.arcLength(contour_convex, True)
				approx = cv2.approxPolyDP(contour_convex, epsilon, False)
				_approx = approx.copy()
				todraw['approxes_bad_before_fixes'].append(_approx)
				# print(approx)
				approx = self.fix_corners(approx)
				# print(approx)
				approx = self.fix_angles(approx)
				# print(approx)
				# img_in = cv2.drawContours(img_in, [approx],  -1, (0, 0, 255), thickness=3)
				answer, nvxt_ = self.make_answer(approx)
				answers.append(answer)
				todraw['contours'].append(contour)
				if answer != None:
					todraw['approxes'].append(approx)
					nvxt.append(nvxt_)
				else:
					todraw['approxes_bad_after_fixes'].append(approx)
		answers = [x for x in answers if x != None]
		return answers, todraw, nvxt

	def draw_corners(self, disp_image, todraw, answers):
		img_out = disp_image
		# for contour in todraw['contours']:
		# 	img_out = cv2.drawContours(img_out, [contour],  -1, (0, 255, 0), thickness=3, offset=(self.Y1,self.X1))
		# for approx in todraw['approxes_bad_before_fixes']:
		# 	img_out = cv2.drawContours(img_out, [approx ],  -1, (255, 255, 0), thickness=4, offset=(self.Y1,self.X1))
		# for approx in todraw['approxes']:
		# 	img_out = cv2.drawContours(img_out, [approx ],  -1, (0, 0, 255), thickness=2, offset=(self.Y1,self.X1))
		# for approx in todraw['approxes_bad_after_fixes']:
		# 	img_out = cv2.drawContours(img_out, [approx ],  -1, (255, 0, 0), thickness=2, offset=(self.Y1,self.X1))
		# print(answers)
		#print(f"answers = {answers}")
		for i in answers:
			for k in i.keys():
				# ~ if k == 7:
					# ~ continue
				v = i[k][0]
				img_out = cv2.circle(img_out, (int(v[0][0]+self.Y1), int(v[0][1])), 6, (255, 0, 255), 3)
		return img_out


	@staticmethod
	def return_number_of_vertexes(answers):
		number_of_vertexes = [8]*len(answers)
		for itr, i in enumerate(answers):
			for k in i.keys():
				#print(f'len_i = {len(i)}')
				if len(i)==4:
					number_of_vertexes[itr] = 4
					continue
				for t in range(k+1, len(i)+1):

					v1 = i[k][0]
					v2 = i[t][0]
					if abs(v1[0][0] - v2[0][0]) <=10 and abs(v1[0][1] - v2[0][1]) <=10:
						number_of_vertexes[itr] -= 1
						break
			if itr > 2 and number_of_vertexes[itr] == 8:
				break

		return number_of_vertexes


	# @staticmethod
	# def rotate_answers(answers):
	# 	new_answer = []
	# 	for i in answers:
	# 		for k in i.keys():
	# 			v = i[k][0]
	# 			#img_out = cv2.circle(img_out, (v[0][0]+self.Y1, v[0][1]), 6, (255, 0, 255), 3)
	# 			v = [v[0][0]+self.Y1, v[0][1]]
	# 	return answers

	#---------------------------------------------------------------------------------------------

	def find_edges_by_bkg(self, img):
		bkg = self.back_ground_image[self.X1:self.X2, self.Y1:self.Y2]
		# img = img[X1:X2, Y1:Y2]
		bkg_ = cv2.cvtColor(bkg, cv2.COLOR_RGB2GRAY)
		img_ = img
		diff = cv2.absdiff(img_, bkg_)
		_, diff = cv2.threshold(diff, 60, 250, cv2.THRESH_TOZERO)
		return diff

	@staticmethod
	def dist(a, b):
		return ((a[0][0]-b[0][0])**2 + (a[0][1]-b[0][1])**2)**0.5

	def make_answer_rectangle(self, approx):
		i1, i2, i3, i4 = approx[0], approx[1], approx[2], approx[3]
		a = self.dist(i1, i2)
		b = self.dist(i2, i3)
		if a>b:
			i1, i2, i3, i4 = approx[1], approx[2], approx[3], approx[0]
			a = self.dist(i1, i2)
			b = self.dist(i2, i3)
		cos_alpha = b/(a*(2.)**0.5)
		if cos_alpha > 1.0:
			# bad rectangle
			return {
				1: (i1, 4, 2),
				2: (i2, 1, 3),
				3: (i3, 2, 4),
				4: (i4, 3, 1)
			}
		x = (2**0.5) * a * np.sin(np.arccos(cos_alpha))
		# print('cos_alpha=',cos_alpha)
		# print('x=',x)
		v23 = [(i3[0][0]-i2[0][0])/self.dist(i3, i2), (i3[0][1]-i2[0][1])/self.dist(i3, i2)]
		i5 = np.array([[round((i1[0][0]+i4[0][0])/2. - x*v23[0]/2.), round((i1[0][1]+i4[0][1])/2. - x*v23[1]/2.)],])
		i6 = np.array([[round((i2[0][0]+i3[0][0])/2. - x*v23[0]/2.), round((i2[0][1]+i3[0][1])/2. - x*v23[1]/2.)],])
		i7 = np.array([[round((i1[0][0]+i4[0][0])/2. + x*v23[0]/2.), round((i1[0][1]+i4[0][1])/2. + x*v23[1]/2.)],]) # consider it the hidden one
		i8 = np.array([[round((i2[0][0]+i3[0][0])/2. + x*v23[0]/2.), round((i2[0][1]+i3[0][1])/2. + x*v23[1]/2.)],]) # consider it the hidden one
		answer = {
			1: (i1, 2, 5, 7),
			2: (i2, 1, 6, 8),
			3: (i3, 4, 6, 8),
			4: (i4, 3, 5, 7),
			5: (i5, 1, 4, 6),
			6: (i6, 2, 3, 5),
			7: (i7, 1, 4, 8),
			8: (i8, 2, 3, 7)
		}
		return answer

	def make_answer(self, approx):
		# number of vertices
		nvxt = len(approx)
		# print('nvxt = ', nvxt)
		if nvxt == 4: # Rectangle
			answer = self.make_answer_rectangle(approx)
		elif nvxt == 6: # Hexagon
			i1, i2, i3, i4, i5, i6 = approx[0], approx[1], approx[2], approx[3], approx[4], approx[5]
			v1 = [i2[0][0]-i1[0][0], i2[0][1]-i1[0][1]]
			i7 = np.array([[i3[0][0]-v1[0], i3[0][1]-v1[1]],])
			v2 = [i3[0][0]-i2[0][0], i3[0][1]-i2[0][1]]
			i8 = np.array([[i4[0][0]-v2[0], i4[0][1]-v2[1]],]) # consider it the hidden one
			answer = {
				1: (i1, 4, 2, 8),
				2: (i2, 1, 3, 7),
				3: (i3, 2, 4, 8),
				4: (i4, 3, 5, 7),
				5: (i5, 4, 6, 8),
				6: (i6, 5, 1, 7),
				7: (i7, 2, 4, 6),
				8: (i8, 1, 3, 5)
			}
		else:
			answer = None
		return answer, nvxt

	@staticmethod
	def find_angle(a, b, c):
		v1 = [b[0][0]-a[0][0], b[0][1]-a[0][1]]
		v2 = [c[0][0]-b[0][0], c[0][1]-b[0][1]]
		dot = v1[0]*v2[0] + v1[1]*v2[1]
		cos = dot/(((v1[0]**2 + v1[1]**2)**0.5) * ((v2[0]**2 + v2[1]**2)**0.5))
		return np.degrees(np.arccos(cos))

	
	def fix_angles(self, approx):
		if len(approx)<3:
			return approx
		# shift
		for i in range(len(approx)-1):
			angle = self.find_angle(approx[i-1], approx[i], approx[i+1])
			# print('a=',angle)
			if angle > self.MINANGLE:
				shift = i
		_approx = []
		for i in range(len(approx)):
			if i >= shift:
				_approx.append([[approx[i][0][0],approx[i][0][1]],])
		for i in range(len(approx)):
			if i < shift:
				_approx.append([[approx[i][0][0],approx[i][0][1]],])
		approx = np.array(_approx)
		#remove
		approx_upd = approx
		for j in range(len(approx)):
			_approx_upd = []
			noremoves = True
			for i in range(len(approx_upd)-1):
				if noremoves:
					angle = self.find_angle(approx_upd[i-1], approx_upd[i], approx_upd[i+1])
					if angle < self.MINANGLE:
						noremoves = False
					else:
						_approx_upd.append([[approx_upd[i][0][0],approx_upd[i][0][1]],])
				else:
					_approx_upd.append([[approx_upd[i][0][0],approx_upd[i][0][1]],])
			if noremoves:
				angle = self.find_angle(approx_upd[-2], approx_upd[-1], approx_upd[0])
				if angle > self.MINANGLE:
					_approx_upd.append([[approx_upd[-1][0][0],approx_upd[-1][0][1]],])
			approx_upd = np.array(_approx_upd)
		return approx_upd

	
	def fix_corners(self, approx):
		#shift
		for i in range(len(approx)-1):
			d = ((approx[i][0][0]-approx[i+1][0][0])**2 + (approx[i][0][1]-approx[i+1][0][1])**2)**0.5
			if d > self.MINDIST:
				shift = i
		_approx = []
		for i in range(len(approx)):
			if i >= shift:
				_approx.append([[approx[i][0][0],approx[i][0][1]],])
		for i in range(len(approx)):
			if i < shift:
				_approx.append([[approx[i][0][0],approx[i][0][1]],])
		approx = np.array(_approx)
		#remove
		approx_upd = approx
		for j in range(len(approx)):
			# print('------------- len(approx_upd) = ',len(approx_upd))
			_approx_upd = []
			noremoves = True
			for i in range(len(approx_upd)-1):
				if noremoves:
					d = ((approx_upd[i][0][0]-approx_upd[i+1][0][0])**2 + (approx_upd[i][0][1]-approx_upd[i+1][0][1])**2)**0.5
					# print('d=',d)
					if d < self.MINDIST:
						noremoves = False
					else:
						_approx_upd.append([[approx_upd[i][0][0],approx_upd[i][0][1]],])
				else:
					_approx_upd.append([[approx_upd[i][0][0],approx_upd[i][0][1]],])
			if noremoves:
				d = ((approx_upd[0][0][0]-approx_upd[-1][0][0])**2 + (approx_upd[0][0][1]-approx_upd[-1][0][1])**2)**0.5
				# print('dd=',d)
				if d > self.MINDIST:
					_approx_upd.append([[approx_upd[-1][0][0],approx_upd[-1][0][1]],])
					# print(_approx_upd)
			else:
				_approx_upd.append([[approx_upd[-1][0][0],approx_upd[-1][0][1]],])
			approx_upd = np.array(_approx_upd)
			# print(_approx_upd)
		return approx_upd








