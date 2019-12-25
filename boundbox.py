import numpy as np

class BoundBox:
	def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
		self.xmin = xmin
		self.ymin = ymin
		self.xmax = xmax
		self.ymax = ymax
		
		self.c     = c
		self.classes = classes

		self.label = -1
		self.score = -1

	def get_label(self):
		if self.label == -1:
			self.label = np.argmax(self.classes)
		
		return self.label
	
	def get_score(self):
		if self.score == -1:
			self.score = self.classes[self.get_label()]
			
		return self.score

	def get_center(self):
		return (self.xmin + (self.xmax - self.xmin)/ 2), (self.ymin + (self.ymax - self.ymin)/ 2)