from abc import ABC, abstractmethod
import PySimpleGUI as sg
import string

# Базовый класс интерфейса
class baseInterface(ABC):

	def __init__(self, themeStyle, TextForApp):
		self.sg = sg
		self.sg.theme(themeStyle)
		self.icon = "data/8.ico"
		# Словарь букв для отображения линий
		self.letter_dict = dict(zip([i for i in range(0,26)],string.ascii_uppercase))
		layout = []
		self.window = self.sg.Window(TextForApp, 
						layout, icon=self.icon, resizable=True)
		
		# Methods from another places
		# ...
		

	@abstractmethod
	def run(self):
		# The PSG "Event Loop"
		while True:
			# Get events for the window with 20ms max wait
			event, values = self.window.Read(timeout=20, timeout_key='timeout')

			if event is None or event == self.sg.WIN_CLOSED or event == 'Cancel':  
				break   
