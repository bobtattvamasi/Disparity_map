import cv2
import os
import numpy as np
from config.config import configValues

class VertexFinder:
	def __init__(self):
		self.MINDIST = 30
		self.MINANGLE = 20
		self.Y1, self.Y2, self.X1, self.X2 = 367, 1000, 0, 700
		self.back_ground_image = cv2.imread('data/BACKGROUND_IMAGE/bkg3.png')
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
				answer = self.make_answer(approx)
				answers.append(answer)
				todraw['contours'].append(contour)
				if answer != None:
					todraw['approxes'].append(approx)
				else:
					todraw['approxes_bad_after_fixes'].append(approx)
		answers = [x for x in answers if x != None]
		return answers, todraw, approx

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
				img_out = cv2.circle(img_out, (v[0][0]+self.Y1, v[0][1]), 6, (255, 0, 255), 3)
		return img_out

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
		#bkg = cv2.imread("./input/bkg1.png")
		# bkg = cv2.imread("./input/bkg2.png")

		bkg = self.back_ground_image[self.X1:self.X2, self.Y1:self.Y2]
		# img = img[X1:X2, Y1:Y2]
		# cv2.imwrite("./output/_bkg.png", bkg)
		# cv2.imwrite("./output/_img.png", img)
		bkg_ = cv2.cvtColor(bkg, cv2.COLOR_RGB2GRAY)
		#img_ = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img_ = img
		diff = cv2.absdiff(img_, bkg_)
		# cv2.imwrite("./output/diff.png", diff)
		# _, diff = cv2.threshold(diff, 100, 250, cv2.THRESH_TOZERO)
		_, diff = cv2.threshold(diff, 60, 250, cv2.THRESH_TOZERO)
		# cv2.imwrite("./output/diff2.png", diff)
		# cv2.imwrite("./output/orig.png", img_)
		return diff

	@staticmethod
	def make_answer(approx):
		# number of vertices
		nvxt = len(approx)
		# print('nvxt = ', nvxt)
		if nvxt == 4: # Rectangle
			i1, i2, i3, i4 = approx[0], approx[1], approx[2], approx[3]
			answer = {
				1: (i1, 4, 2),
				2: (i2, 1, 3),
				3: (i3, 2, 4),
				4: (i4, 3, 1)
		}
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
		return answer

	@staticmethod
	def find_angle(a, b, c):
		v1 = [b[0][0]-a[0][0], b[0][1]-a[0][1]]
		v2 = [c[0][0]-b[0][0], c[0][1]-b[0][1]]
		dot = v1[0]*v2[0] + v1[1]*v2[1]
		cos = dot/(((v1[0]**2 + v1[1]**2)**0.5) * ((v2[0]**2 + v2[1]**2)**0.5))
		# print('cos=',cos)
		# print('degrees=',np.degrees(np.arccos(cos)))
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








