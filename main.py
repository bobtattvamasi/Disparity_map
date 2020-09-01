import sys
if len(sys.argv)>1:
	from GUI.interface_no_pi import Interface
else:
	from GUI.interface import Interface

def main():
	interbox = Interface("DarkAmber", 'Demo Determine Line Lenght')
	interbox.run()
	interbox.close()

if __name__ == '__main__':
	main()
