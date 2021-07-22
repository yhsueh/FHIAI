import numpy as np

class Point():
	def __init__(self, pt):
		self.x = pt[0]
		self.y = pt[1]

	def get_point(self):
		return self

	def get_point_tuple(self):
		return((self.x, self.y))

	def get_distance(self, pt):
		return np.sqrt((self.x - pt.x)**2 + (self.y - pt.y)**2)

	def get_projected_line(self, line):
		slope = line.get_slope()
		intercept = self.y - slope*self.x
		return slope, intercept

	def add_point(self, point):
		self.x += point[0]
		self.y += point[1]
		return self

	def switch_coordinate_system(self, img):
		# OpenCV image coordinate:| Math coordinate:    
		# ----->x                 | y
		# |                       | ^ 
		# |                       | |
		# v                       | |
		# y                       | |------>x
		convert_pt = (self.x, img.shape[0] - self.y)
		return Point(convert_pt)

class Line():
	# Assmption: All the coordinates in this class use math coordinate system.
	def __init__(self, pt1, pt2):
		assert type(pt1) is Point and type(pt2) is Point, "Points are of wrong type."
		self.pt1 = pt1
		self.pt2 = pt2
		self.__compute()

	def __compute(self):
		self.slope = (self.pt2.y - self.pt1.y) / (self.pt2.x - self.pt1.x)
		self.intercept = self.pt1.y - self.slope*self.pt1.x

	def calculate_intersection(self, slope, intercept):
		x_intersect = (intercept - self.intercept)/(self.slope - slope)
		y_intersect = self.slope*x_intersect + self.intercept
		return Point((x_intersect, y_intersect))

	def get_slope(self):
		return self.slope

	def get_intercept(self):
		return self.intercept