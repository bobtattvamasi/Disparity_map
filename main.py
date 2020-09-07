import sys
from GUI.interface import MainWindow

def main():
	interbox = None
	print(len(sys.argv))
	if len(sys.argv)==1:
		interbox = MainWindow("DarkAmber", 'Demo Determine Line Lenght', False) 
	else:
		interbox = MainWindow("DarkAmber", 'Demo Determine Line Lenght', True)
	interbox.run()
	interbox.close()

if __name__ == '__main__':
	main()
