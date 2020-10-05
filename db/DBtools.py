import csv

class DataBase:
	def __init__(self):
		# Main Window's parameters
		self.mWinParameters = self.read_csv("db/settings.csv")
		# autoDetectrectWindow's parameters
		self.aDRWinParameters = self.read_csv("db/secondWin.csv")

	@staticmethod
	def read_csv(file_name):
		parameters = {}
		reader = csv.reader(open(file_name, 'r'))
		for row in reader:
			k,v = row
			
			if k != 'sigmaColor':
				parameters[k] = int(v)
			else:
				parameters[k] = float(v)

		return parameters

	@staticmethod
	def save_csv(file_name, parameters):
		with open(file_name,"w") as f:
			w = csv.writer(f)
			for key, val in parameters.items():
				w.writerow([key, val])
		print(f"settings saved in {file_name}")

	def update_mWinParameters(self, values):

		self.mWinParameters['SpklWinSze'] = int(values[0])
		self.mWinParameters['SpcklRng'] = int(values[1])
		self.mWinParameters['UnicRatio'] = int(values[2])
		self.mWinParameters['TxtrThrshld'] = int(values[3])
		self.mWinParameters['NumOfDisp'] = int(values[4]/16)*16
		self.mWinParameters['MinDISP'] = int(values[5])
		self.mWinParameters['PreFiltCap'] = int(values[6]/2)*2+1
		self.mWinParameters['PFS'] = int(values[7]/2)*2+1
		# ~ self.mWinParameters['lambda'] = int(values[8])
		# ~ self.mWinParameters['sigmaColor'] = float(values[9])
		# ~ self.mWinParameters['Radius'] = int(values[10])
		# ~ self.mWinParameters['LRCthresh'] = int(values[11])

	def update_aDRWinParameters(self, values):

		self.aDRWinParameters["lowH"] = int(values[0])
		self.aDRWinParameters["highH"] = int(values[1])
		self.aDRWinParameters["lowS"] = int(values[2])
		self.aDRWinParameters["highS"] = int(values[3])
		self.aDRWinParameters["lowV"] = int(values[4])
		self.aDRWinParameters["highV"] = int(values[5])

		self.aDRWinParameters["minVal"] = int(values[6])
		self.aDRWinParameters["maxVal"] = int(values[7])
		self.aDRWinParameters["layer"] = int(values[8])
		self.aDRWinParameters["Thmin"] = int(values[9])
		self.aDRWinParameters["Thmax"] = int(values[10])


