
def change_params(params, imageToDisp):
	imgL, imgR = calibrate_two_images(imageToDisp)
	rectified_pair = (imgL, imgR)
	
	self.parameters['SpklWinSze'] = int(values[0])
	self.parameters['SpcklRng'] = int(values[1])
	self.parameters['UnicRatio'] = int(values[2])
	self.parameters['TxtrThrshld'] = int(values[3])
	self.parameters['NumOfDisp'] = int(values[4]/16)*16
	self.parameters['MinDISP'] = int(values[5])
	self.parameters['PreFiltCap'] = int(values[6]/2)*2+1
	self.parameters['PFS'] = int(values[7]/2)*2+1

	disparity, value_disparity = stereo_depth_map(rectified_pair, self.parameters)