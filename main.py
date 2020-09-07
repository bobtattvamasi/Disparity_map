import sys
# if len(sys.argv)>1:
# 	from GUI.interface_no_pi import MainWindow
# else:
from GUI.interface import MainWindow

def main():
	interbox = MainWindow("DarkAmber", 'Demo Determine Line Lenght', False) #if len(sys.argv)==0 else interbox = MainWindow("DarkAmber", 'Demo Determine Line Lenght', ifCamPi=False)
	interbox.run()
	interbox.close()

if __name__ == '__main__':
	main()
