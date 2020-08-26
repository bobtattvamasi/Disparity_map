import csv

def read_csv(file_name):
	parameters = {}
	reader = csv.reader(open(file_name, 'r'))
	for row in reader:
		k,v = row
		parameters[k] = int(v)

	return parameters

def save_scv(file_name, parameters):
	with open(file_name,"w") as f:
		w = csv.writer(f)
		for key, val in parameters.items():
			w.writerow([key, val])
	print(f"settings saved in {file_name}")
