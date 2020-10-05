import cv2
from typing import NamedTuple
import numpy as np
class configValues(NamedTuple):
	PHOTO_WIDTH = 2560 #3840
	PHOTO_HEIGHT = 720 #1088
	IMAGE_WIDTH = 1280 #1920
	IMAGE_HEIGHT = 720 #1088
	WINDOW_WIDTH = 640
	WINDOW_HEIGHT = 362
	#folderTestScenes = './data/scenes4test/*.png'
	#folderTestScenes = './data/objects_high/*.png'
	#folderTestScenes = './data/objects_low/*.png'
	#folderTestScenes = './data/photo_data/*.png'
	folderTestScenes = './data/test_disparity/*.png'
	folderTestScenes = './data/NEW_IMAGES/*.png'
	
photo_width = 2560
photo_height = 720
image_width = 1280
image_height = 720
image_size = (image_width,image_height)

configValues = configValues()
# Функция калибрует два изображения, разделяет их и эти два возвращает
def calibrate_two_images(imageToDisp, ifCamPi = True, calibratefolder = './data'):
	# Определяем размеры(ширина,высота) сдвоенного изображения со стереокамеры и
	# разделенных левого и правого.
	

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
		npzfile = np.load(f'{calibratefolder}/calibration_data/{image_height}p/stereo_camera_calibration.npz')
	except:
		print("Camera calibration data not found in cache, file " & f'{calibratefolder}/calibration_data/{image_height}p/stereo_camera_calibration.npz')
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

imagetodisp = '../data/NEW_IMAGES/scene_2560x720_1.png'

pair_img = cv2.imread(imagetodisp,0)
imgLeft = pair_img [0:photo_height,0:image_width] #Y+H and X+W

rectified_pair = calibrate_two_images(imagetodisp, ifCamPi=False,calibratefolder='../data')

dmLeft = rectified_pair[0]
dmRight = rectified_pair[1]

sbm = cv2.StereoBM_create(numDisparities=16, blockSize=15)

sbm.setPreFilterType(1)
sbm.setPreFilterSize(5)
sbm.setPreFilterCap(63)
sbm.setMinDisparity(254)
sbm.setNumDisparities(80)
sbm.setTextureThreshold(0)
sbm.setUniquenessRatio(14)
sbm.setSpeckleRange(0)
sbm.setSpeckleWindowSize(1)

disparity0 = sbm.compute(dmLeft, dmRight)

cv2.namedWindow('filteredimage')

# Disparity_filter
#--------------------------------------------------------------------------------
# wls_filter = cv2.ximgproc.createDisparityWLSFilter(sbm)
# right_matcher = cv2.ximgproc.createRightMatcher(sbm)
# wls_filter = cv2.ximgproc.createDisparityWLSFilter(sbm)
# right_matcher = cv2.ximgproc.createRightMatcher(sbm)
# right_disp = right_matcher.compute(dmRight, dmLeft)
# wls_filter.setLambda(8000)	
# wls_filter.setSigmaColor(1.5)
# wls_filter.setDepthDiscontinuityRadius(5)
# wls_filter.setLRCthresh(24)
# disparity = wls_filter.filter(disparity, dmLeft, disparity_map_right=right_disp )
#--------------------------------------------------------------------------------

#AdaptiveManifoldFilter
#--------------------------------------------------------------------------------
#adaptive_filter	= cv2.ximgproc.AdaptiveManifoldFilter_create()

def nothing(x):
    pass


# create trackbars for color change
#cv2.createTrackbar('AdjustOutliers','filteredimage',0,255,nothing)
#cv2.createTrackbar('PCAIterations','filteredimage',0,255,nothing)
cv2.createTrackbar('SigmaR','filteredimage',0,255,nothing) # /10
cv2.createTrackbar('SigmaS','filteredimage',0,255,nothing) # /10
cv2.createTrackbar('TreeHeight','filteredimage',-255,255,nothing)
cv2.createTrackbar('UseRNG','filteredimage',0,1,nothing)
#disparity	= adaptive_filter.filter(disparity0)
#--------------------------------------------------------------------------------

# 
#---

#----

def create_disparity_image(disparity):
	local_max = disparity.max()
	local_min = disparity.min()
	autotune_max = local_max
	autotune_min = local_min

	disparity_grayscale = (disparity-autotune_min)*(65535.0/(autotune_max-autotune_min))
	disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
	disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)

	return disparity_color

disparity_color = create_disparity_image(disparity0)

#cv2.imshow('grayscale', disparity_grayscale)

original_disparity = create_disparity_image(disparity0)

disparity = cv2.ximgproc_DTFilter.filter(original_disparity)

while(1):


	cv2.imshow("original disparity", original_disparity)
	cv2.imshow("filteredimage", disparity_color)
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break

	# get current positions of four trackbars
	#AdjustOutliers = cv2.getTrackbarPos('AdjustOutliers','filteredimage')
	#PCAIterations = cv2.getTrackbarPos('PCAIterations','filteredimage')
	SigmaR = cv2.getTrackbarPos('SigmaR','filteredimage')
	TreeHeight = cv2.getTrackbarPos('TreeHeight','filteredimage')
	SigmaS = cv2.getTrackbarPos('SigmaS','filteredimage')
	UseRNG = cv2.getTrackbarPos('UseRNG','filteredimage')

	

	#adaptive_filter.setAdjustOutliers(AdjustOutliers)
	#adaptive_filter.setPCAIterations(PCAIterations)
	# adaptive_filter.setSigmaR(SigmaR/10.0)
	# adaptive_filter.setSigmaS(SigmaS/10.0)
	# adaptive_filter.setTreeHeight(TreeHeight)
	# adaptive_filter.setUseRNG(UseRNG)

	disparity	= cv2.ximgproc_DTFilter.filter(disparity0)
	disparity_color = create_disparity_image(disparity)


cv2.destroyAllWindows()

	 
