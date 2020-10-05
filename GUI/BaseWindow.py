from abc import ABC, abstractmethod
import PySimpleGUI as sg
import string

from db.DBtools import DataBase

# Базовый класс интерфейса
class BaseWindow(ABC):

	def __init__(self, themeStyle, TextForApp):
		self.sg = sg
		# Тема. По дефолту используется Dark
		self.sg.theme(themeStyle)
		# Икона для окна, которая пока не подгружается
		self.icon = "data/8.ico"
		# Словарь букв для отображения линий
		self.letter_dict = dict(zip([i for i in range(0,26)],string.ascii_uppercase))
		layout = []
		self.window = self.sg.Window(TextForApp, 
						layout, icon=self.icon, resizable=True)

		# Инициализация работы с базой данных
		self.db = DataBase()

		# Для выбора языка
		# 0 - английский
		# 1 - русский
		self.language = 0
		
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
