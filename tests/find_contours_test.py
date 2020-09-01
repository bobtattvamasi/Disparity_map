import cv2
import imutils

img = cv2.imread("thresh.jpg")

image_blur = cv2.medianBlur(img, 25)

img_blur_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)

image_res, image_thresh = cv2.threshold(img_blur_gray, 240,255, cv2.THRESH_BINARY_INV)

cnts = cv2.findContours(image_thresh.copy(), cv2.RETR_EXTERNAL,
						cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)

for i,c in enumerate(cnts):
	cv2.drawContours(img, [c], -1, (0,255,0), 2)

cv2.imshow("image_thresh",image_thresh)
cv2.imshow("image_res",image_res)
cv2.imshow("showTime", img)
cv2.waitKey(0)
