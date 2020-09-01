import cv2
import numpy as np

image = cv2.imread("fulldisp.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

boundaries= [
		#([17,15,100], [50,56,200]),
		([15,17,100],[56,50,200]),
		([86,31,4],[220,88,50]),
		([25,146,190],[62,174,250]),
		([103,86,65],[145,133,128]),
		([8,0,0],[255,4,4])
]

i=0
for (lower, upper) in boundaries:
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	
	# find the colors with specified boundaries and apply the mask
	mask = cv2.inRange(image, lower,upper)
	height,width,_ = image.shape
	black_img = np.zeros((height,width))
	
	second_inRange = cv2.inRange(image, np.array([0,0,0]), np.array([8,8,255]))
	
	output = cv2.bitwise_and(image, image, mask = second_inRange)
	output[output>0] = 255
	#gray[mask>0] = 0
	#gray[mask==0] = 255
	#image[image<255] = 0
	print(image[313,190])
	print(f"i={i} ; lower = {lower}, upper = {upper}")
	
	cv2.imshow("gray", gray)
	cv2.imshow("second", mask)
	cv2.imshow('images', np.hstack([image,output]))
	cv2.waitKey(0)
	i=i+1
