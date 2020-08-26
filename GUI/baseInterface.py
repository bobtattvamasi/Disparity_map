from abc import ABC, abstractmethod
import PySimpleGUI as sg

# Базовый класс интерфейса
class baseInterface(ABC):

	def __init__(self, themeStyle, TextForApp):
		self.sg = sg
		self.sg.theme(themeStyle)
		self.icon = "data/8.ico"
		layout = []
		self.window = self.sg.Window(TextForApp, 
						layout, icon=self.icon, resizable=True)

	@abstractmethod
	def run(self):
		# The PSG "Event Loop"
		while True:
			# Get events for the window with 20ms max wait
			event, values = self.window.Read(timeout=20, timeout_key='timeout')

			if event is None or event == self.sg.WIN_CLOSED or event == 'Cancel':  
				break   