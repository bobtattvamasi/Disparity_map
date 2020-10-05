import sys
import argparse
from GUI.MainWindow import MainWindow

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("-m",  "--mode",	 type=str2bool, required=True,  default=False, help="with cameraPi or Pictures")
	ap.add_argument("-d", "--debugmode",  type=str2bool, required=False,  default=False, help="In debugmode right picture always rewrite disparity map")
	args = vars(ap.parse_args())

	mainWindow = MainWindow("DarkAmber", 'Demo Determine Line Lenght', args['mode'])
	mainWindow.run(ifDebugVersion=args['debugmode'])
	mainWindow.close()

def str2bool(v):
	if isinstance(v, bool):
	   return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
	main()
