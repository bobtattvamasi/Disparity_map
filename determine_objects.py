import cv2, PySimpleGUI as sg
from determine_helper import convert_to_bytes, calibrate_two_images, stereo_depth_map, contours_finder
import base64
import glob
import sys, traceback
from point_cloud import stereo_depth_map as depth_map, create_points_cloud

sg.theme('DarkAmber')

#------------------------------------------
photo_files = []
n=0
for file in glob.glob('./scenes4test/*.png'):
#for file in glob.glob('./kubic/*.png'):
	photo_files.append(file)
	n = n+1

i = 0
# imageToDisp = './scenes4test/scene_3840x1088_20.png'
imageToDisp = photo_files[i]

parameters = {'SpklWinSze':0,
				'SpcklRng': 0,
				'UnicRatio':25,
				'TxtrThrshld':0,
				'NumOfDisp':80,
				'MinDISP':61,
				'PreFiltCap':5,
				'PFS': 5,
				'SWS': 5}


#------------------------------------------

left_col = [
				[sg.Image(filename='', key='image')],
				#[sg.Text(size=(25,3), key='-OUT-')],

				[sg.Text("SpklWinSze"), sg.Slider(range=(0, 300), orientation='h', size=(34, 10), default_value=100)],
				[sg.Text("SpcklRng"), sg.Slider(range=(0, 40), orientation='h', size=(34, 10), default_value=15)],
				[sg.Text("UnicRatio"), sg.Slider(range=(1, 80), orientation='h', size=(34, 10), default_value=10)],
				[sg.Text("TxtrThrshld"), sg.Slider(range=(0, 1000), orientation='h', size=(34, 10), default_value=100)],
				[sg.Text("NumOfDisp"), sg.Slider(range=(16, 256), orientation='h', size=(34, 10), default_value=128)],
				[sg.Text("MinDISP"), sg.Slider(range=(-300, 300), orientation='h', size=(34, 10), default_value=61)],
				[sg.Text("PreFiltCap"), sg.Slider(range=(5, 63), orientation='h', size=(34, 10), default_value=29)],
				[sg.Text("PFS"), sg.Slider(range=(5, 255), orientation='h', size=(34, 10), default_value=5)],
				[sg.Text("SWS"), sg.Slider(range=(5, 255), orientation='h', size=(34, 10), default_value=5)],

				[sg.Button('Determine', size=(15,2)), sg.Button('Next Picture ->', size=(15,2))]
]

right_col = [
				#[sg.Image(filename='', key='deep_image')],
				[sg.Graph(canvas_size=(700,400),
					graph_bottom_left=(0, 0),
					graph_top_right=(700, 400),
					key="graph",
					change_submits=True, # mouse click events
					drag_submits=True # mouse drag events]
					)],
				[sg.Output(size=(64, 12))],

				[sg.Text("minVal"), sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=50)],
				[sg.Text("maxVal"), sg.Slider(range=(0, 255), orientation='h', size=(34, 10), default_value=200)]
]

layout  = [[sg.Column(left_col, element_justification='c'), sg.VSeperator(),sg.Column(right_col, element_justification='c')]]

window = sg.Window('Demo Determine Dimensions', layout, icon="8.ico")


#cap = cv2.VideoCapture(0)       # Setup the camera as a capture device

imgL, imgR = calibrate_two_images(imageToDisp)


rectified_pair = (imgL, imgR)

disparity = stereo_depth_map(rectified_pair, parameters)
disparity = contours_finder(disparity, 50, 200)

# print(f'disparity = {disparity}')
# disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)


# cv2.imshow("d", disparity)
# cv2.waitKey(0)

# im64 = base64.b64encode(imgL)

a_id = None
graph = window.Element("graph")

dragging = False
start_point, end_point = None, None
start_p, end_p = None, None

while True:                     # The PSG "Event Loop"
	event, values = window.Read(timeout=20, timeout_key='timeout')      # get events for the window with 20ms max wait

	if event is None or event == sg.WIN_CLOSED or event == 'Cancel':  
		break                                            # if user closed window, quit

	try:
		parameters['SpklWinSze'] = int(values[0])
		parameters['SpcklRng'] = int(values[1])
		parameters['UnicRatio'] = int(values[2])
		parameters['TxtrThrshld'] = int(values[3])
		parameters['NumOfDisp'] = int(values[4]/16)*16
		parameters['MinDISP'] = int(values[5])
		parameters['PreFiltCap'] = int(values[6]/2)*2+1
		parameters['PFS'] = int(values[7]/2)*2+1
		parameters['SWS'] = int(values[8]/2)*2+1

		minVal = int(values[10])
		maxVal = int(values[11])


		disparity = stereo_depth_map(rectified_pair, parameters)
		disparity = contours_finder(disparity, minVal, maxVal)
	
	
		
		#cv2.imwrite("graph.png",  disparity)
		if a_id:
			graph.DeleteFigure(a_id)
		a_id = graph.DrawImage(data=cv2.imencode('.png', disparity)[1].tobytes(), location=(0, 400))
		if start_point is not None and end_point is not None:
			graph.DrawLine(start_point, end_point, color='purple', width=10)
		graph.TKCanvas.tag_lower(a_id)
		#window.FindElement('graph').Update(data="graph.png", location=(0, 400))
		#window.FindElement('image').Update(data=convert_to_bytes(imageToDisp, resize=None)) # Update image in window
		window.FindElement('image').Update(data=cv2.imencode('.png', rectified_pair[0])[1].tobytes()) # Update image in window
		#window.FindElement('deep_image').Update(data=cv2.imencode('.png', disparity)[1].tobytes()) # Update image in window
		# window.FindElement('image').Update(data=im64) # Update image in window
		#window.FindElement('deep_image').Update(data=cv2.imencode('.png', cap.read()[1])[1].tobytes()) # Update image in window
		



	except IndexError:

		print(traceback.format_exc())


	

	if event is "Next Picture ->":
		i = i + 1
		if i >= n:
			i=0
		imageToDisp = photo_files[i] 
		imgL, imgR = calibrate_two_images(imageToDisp)
		

		rectified_pair = (imgL, imgR)

	if event == "graph":
		x,y = values["graph"]
		#print (f"mouse down at ({x},{y})")
		if not dragging:
			start_p = (x,y)
			dragging = True
		else:
			end_p = (x,y)
			
	elif event.endswith('+UP'):
		start_point = start_p
		end_point = end_p
		print(f"grabbed rectangle from {start_point} to {end_point}")
		start_p, end_p = None, None
		dragging = False

	if event is 'Determine':
		disparity2, points_3, colors = create_points_cloud(imageToDisp, parameters)
		print(f'Info: Filepaths correctly defined.{values[0]}')

		



