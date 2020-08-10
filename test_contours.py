import sys
import numpy as np
import cv2 as cv
import math

hsv_min = np.array((2, 28, 65), np.uint8)
hsv_max = np.array((26, 238, 255), np.uint8)

def lines_angle(line1, line2):
    v_a = np.array(line1[1]) - np.array(line1[0])
    v_b = np.array(line2[1]) - np.array(line2[0])

    return np.degrees(np.arccos(np.dot(v_a,v_b) / (np.linalg.norm(v_a) * np.linalg.norm(v_b))))


def find_angle(line1, line2):
    x11, y11 = line1[0]
    x12, y12 = line1[1]

    x21, y21 = line2[0]
    x22, y22 = line2[1]

    k1 = (y12 - y11)/(x12 - x11)
    k2 = (y22 - y21)/(x22 - x21)

    tang_phy = (k2 - k1)/(1+ k1*k2)
    return math.atan(tang_phy)

if __name__ == '__main__':
    fn = 'kubic/9.png'
    img = cv.imread(fn)
    img = cv.resize(img,(680,480))

    #hsv = cv.cvtColor( img, cv.COLOR_BGR2HSV )
    # thresh = cv.inRange( hsv, hsv_min, hsv_max )
    img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    

    index = 0
    layer = 0
    minVal = 50
    maxVal = 140
    isPrint = 0
    contours0 = []
    screenCnt = []
    procent = 0.00

    def update():
        global contours0
        global screenCnt
        vis = img.copy()
        edges = cv.Canny(img1,minVal,maxVal)
        contours0, hierarchy = cv.findContours( edges.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        screenCnt = []
        for c in contours0:
            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, procent * peri, True)
            screenCnt.append(approx)

        cv.drawContours( vis, screenCnt, index, (0,255,0), 2, cv.LINE_AA, hierarchy, layer )
        cv.imshow('contours', vis)

    def update_index(v):
        global index
        index = v-1
        update()

    def update_layer(v):
        global layer
        layer = v
        update()

    def update_minval(v):
        global minVal
        minVal = v
        update()

    def update_maxval(v):
        global maxVal
        maxVal = v
        update()

    def update_print(v):
        global isPrint
        global contours0
        global screenCnt
        isPrint = v
        if v == 1:
            print(f"len = {len(screenCnt)}")
            # print(f"contours = {screenCnt}")
            for c in screenCnt:
                all(print(lines_angle(c[0][i], c[0][i+1])) for i in range(len(c[0][0])))
            v = 0
            isPrint = v

    def update_procent(v):
        global procent
        procent = v/1000
        update()

    update_index(0)
    update_layer(0)
    cv.createTrackbar( "contour", "contours", 0, 7, update_index )
    cv.createTrackbar( "layers", "contours", 0, 7, update_layer )
    cv.createTrackbar( "minVal", "contours", 0, 255, update_minval)
    cv.createTrackbar( "maxVal", "contours", 0, 255, update_maxval)
    cv.createTrackbar( "precent_of_peri", "contours", 0, 100, update_procent)
    cv.createTrackbar( "toPrint", "contours", 0, 1, update_print)


    cv.waitKey()
    cv.destroyAllWindows()