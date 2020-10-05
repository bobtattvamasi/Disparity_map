from abc import ABC, abstractmethod

class BaseObject(ABC):
	def __init__(self):
		self.Mat_image = None
		self.Mat_disparity = None
	
	@abstractmethod
	def _crop_object(self, image):
		pass
		
