import cv2
import imutils
import numpy as np

# img1 = cv2.imread("fulldisp.jpg")
# img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# edges = cv2.Canny(img,10,200)

# cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

# hull = cv2.convexHull(cnts[1])



# screenCnt = []
# for c in cnts:
# 	print(f" CONTOUR_Area = {cv2.contourArea(c)}")
# 	contour_area = cv2.contourArea(c)
# 	# approximate the contour
# 	peri = cv2.arcLength(c, True)
# 	print(f"peri={peri}")
# 	approx = cv2.approxPolyDP(c, 0.015 * peri, True)
# 	#print(f"spprox = {approx}")
# 	# if our approximated contour has four points, then
# 	# we can assume that we have found our screen
# 	# if contour_area < 25:
# 	if peri > 490:
# 		screenCnt.append(approx)



# cv2.drawContours(img1, screenCnt, -1, (0, 255,0), 3)

# cv2.imshow("contours", img)
# cv2.imshow("edges", edges)
# cv2.imshow("origin", img1)
# cv2.waitKey(0)

def lines_angle(line1, line2):
	v_a = np.array(line1[1]) - np.array(line1[0])
	v_b = np.array(line2[1]) - np.array(line2[0])

	return np.degrees(np.arccos(np.dot(v_a,v_b) / (np.linalg.norm(v_a) * np.linalg.norm(v_b))))


point = [[[639,  83]], [[448,  23]], [[ 43,  57]], [[ 44, 376]], [[219, 458]], [[626, 404]]]

for i in range(len(point)):
	for j in range(i+1,len(point)-1):
		line1 = [point[i][0],point[j][0]]
		line2 = [point[i][0],point[j+1][0]]
		print(f"i=[i]\nline1 = {line1}, line2 = {line2}")
		print(lines_angle(line1, line2))


