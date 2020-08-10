import math
import numpy as np


def find_angle(line1, line2):
	x11, y11 = line1[0]
	x12, y12 = line1[1]

	x21, y21 = line2[0]
	x22, y22 = line2[1]

	if (x12 - x11) != 0:
		k1 = (y12 - y11)/(x12 - x11)
	else:
		k1 = None

	if (x22 - x21) != 0:
		k2 = (y22 - y21)/(x22 - x21)
	else:
		k2 = None

	if k1 == None and k2 == 0:
		return 90
	elif k1 == 0 and k2 == None:
		return 90
	elif k1 == None and k2 == None:
		return 0

	if (1+k1*k2) != 0:
		angle = math.degrees(math.atan((k2 - k1)/(1+ k1*k2)))
	else:
		angle = 90

	print(f"k1 = {k1}, k2 = {k2}" )
	return angle


def lines_angle(line1, line2):
	v_a = np.array(line1[1]) - np.array(line1[0])
	v_b = np.array(line2[1]) - np.array(line2[0])

	return np.degrees(np.arccos(np.dot(v_a,v_b) / (np.linalg.norm(v_a) * np.linalg.norm(v_b))))

line1 = [[0,0], [3,0]]
line2 = [[6,0], [3,0]]

print(f"angle = {lines_angle(line1, line2)}")