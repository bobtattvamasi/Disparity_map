import sys
from GUI.MainWindow import MainWindow

def main():
	if len(sys.argv)==1:
		mainWindow = MainWindow("DarkAmber", 'Demo Determine Line Lenght', False) 
	else:
		mainWindow = MainWindow("DarkAmber", 'Demo Determine Line Lenght', True)
	mainWindow.run()
	mainWindow.close()

if __name__ == '__main__':
	main()
