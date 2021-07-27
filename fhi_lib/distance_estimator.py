import argparse
import os
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
from functools import cmp_to_key

from fhi_lib.geometry import Point, Line

class DistanceEstimator():
	def __init__(self, img):
		self.img = img
		self.panel_length = 2235
		self.scale_length = 100

	def initialize(self):
		self.__find_scales()
		self.__form_reference_points()
		self.__shift_accessory_coordinate_init()
		print('Estimator initialized')

	def initialize_with_pt(self, pt):
		self.__find_scales()
		self.__form_reference_points()
		self.vertical_pt2 = Point(pt)
		self.__shift_accessory_coordinate_init()
		print('Estimator initialized')

	def display_reference_pts(self, img):
		img = cv2.circle(img, self.origin.get_point_tuple(), 20, (0,0,0), 3)
		img = cv2.circle(img, self.horizontal_pt.get_point_tuple(), 20, (0,255,0), 3)
		img = cv2.circle(img, self.vertical_pt.get_point_tuple(), 20, (255,0,0), 3)
		img = cv2.circle(img, self.vertical_pt2.get_point_tuple(), 20, (255,0,0), 3)
		img = cv2.circle(img, self.origin.get_point_tuple(), 0, (0,0,255), 3)
		img = cv2.circle(img, self.horizontal_pt.get_point_tuple(), 0, (0,0,255), 3)
		img = cv2.circle(img, self.vertical_pt.get_point_tuple(), 0, (0,0,255), 3)
		img = cv2.circle(img, self.vertical_pt2.get_point_tuple(), 0, (0,0,255), 3)
		return img

	def estimate(self, pt_itr):
		img_intersection = self.__shift_accessory_coordinate(pt_itr)
		dist = self.__cross_ratio(img_intersection)
		caption = '{}\n'.format(int(dist))
		return caption

	def __find_scales(self):
		### Image Processing, convert rgb to hsv and find the scale by its color ###
		blur = cv2.GaussianBlur(self.img, (5,5), 0)
		img_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
		#img_threshold =  cv2.inRange(img_hsv, (25,30,100), (45,190,255))
		img_threshold =  cv2.inRange(img_hsv, (25,60,190), (50,130,255))

		morphology_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
		dilation = cv2.dilate(img_threshold, morphology_kernel, iterations=3)
		thresh = cv2.erode(dilation, morphology_kernel, iterations=3)		

		### Crop the image as we know the scale is always on the left half of the image ###
		cropped_thresh = thresh[:, 0:int(thresh.shape[1]/2)]
		contours, _ = cv2.findContours(image=cropped_thresh,
									   mode=cv2.RETR_EXTERNAL,
									   method=cv2.CHAIN_APPROX_SIMPLE)

		### Discard contours that are not quadrilaterals and smaller than 4000 pixels###
		result_contours = {}
		epsilon = 30
		minimal_area = 2000
		for contour in contours:
			contour_area = cv2.contourArea(contour)
			if contour_area > minimal_area:
				hull = cv2.convexHull(contour)
				approxCurve = cv2.approxPolyDP(hull, epsilon, True)
				if len(approxCurve) == 4:
					result_contours.update({contour_area : approxCurve})

		self.__verify_shape(result_contours)
		
		if (len(result_contours) == 2):
			key_list = list(result_contours.keys())
			self.near_scale = result_contours[key_list[0]]
			self.far_scale = result_contours[key_list[1]]
		else:
			print('Contours Found', len(result_contours.keys()))
			raise Exception('Error: More or fewer than two contours are found!')

	def __verify_shape(self, result_contours):
		tolerance = 0.40
		remove_keys = []

		for key in result_contours.keys():
			pts = result_contours[key]
			pts = pts[:,0,:]
			
			pt1 = Point(pts[0])
			pt2 = Point(pts[1])
			pt3 = Point(pts[2])
			pt4 = Point(pts[3])

			dist1_2 = pt1.get_distance(pt2).astype(np.int)
			dist3_4 = pt3.get_distance(pt4).astype(np.int)
			dist1_4 = pt1.get_distance(pt4).astype(np.int)
			dist2_3 = pt2.get_distance(pt3).astype(np.int)

			if np.absolute(dist1_2 - dist3_4) / np.min([dist1_2, dist3_4])> tolerance:
				remove_keys.append(key)
				continue
			elif np.absolute(dist1_4 - dist2_3) / np.min([dist1_4, dist2_3])> tolerance:
				remove_keys.append(key)
				continue

		for remove_key in remove_keys:
			del result_contours[remove_key]

	def __form_reference_points(self):
		### Get two points with smallest x-coordinates ###
		self.near_scale = self.near_scale[:,0,:]
		self.far_scale = self.far_scale[:,0,:]

		self.far_scale = self.__set_orientation_hull(self.far_scale)
		self.near_scale = self.__set_orientation_hull(self.near_scale)
				
		self.origin = Point(self.near_scale[1])
		self.vertical_pt = Point(self.near_scale[0])
		self.horizontal_pt = Point(self.near_scale[2])
		self.vertical_pt2 = Point(self.far_scale[0])

	def __set_orientation_approx(self, scale):
		# Assuming the scale is placed on the left half of the image.
		# The first vertex should be top left. If it's not the case, then reorder the verticies.
		order = scale[:,0].argsort()
		if order[0].astype(int) == 0:
			## 3 2 ##
			## 0 1 ##
			# The first vertex is at bottom left instead of top left. Reorder the verticies.
			scale = scale[[3,0,1,2]]
		elif order[0].astype(int) == 2:
			## 1 0 ##
			## 2 3 ##
			scale = scale[[1,2,3,0]]
		elif order[0].astype(int) == 3:
			## 2 1 ##
			## 3 0 ## 
			scale = scale[[2,3,1,0]]
		return scale

	def __set_orientation_hull(self, scale):
		# Assuming the scale is placed on the left half of the image.
		# The first vertex should be top left. If it's not the case, then reorder the verticies.
		order = scale[:,0].argsort()
		if order[0].astype(int) == 0:
			## 1 2 ##
			## 0 3 ##
			# The first vertex is at bottom left instead of top left. Reorder the verticies.
			scale = scale[[1,0,3,2]]
		elif order[0].astype(int) == 1:
			## 2 3 ##
			## 1 0 ##
			# The first vertex is at bottom left instead of top left. Reorder the verticies.
			scale = scale[[2,1,0,3]]
		elif order[0].astype(int) == 2:
			## 3 0 ##
			## 2 1 ##
			scale = scale[[3,2,1,0]]
		elif order[0].astype(int) == 3:
			## 0 1 ##
			## 3 2 ## 
			scale = scale[[0,3,2,1]]
		return scale

	def __shift_accessory_coordinate_init(self):
		math_origin = self.origin.switch_coordinate_system(self.img)
		math_horizontal_pt = self.horizontal_pt.switch_coordinate_system(self.img)
		math_vertical_pt2 = self.vertical_pt2.switch_coordinate_system(self.img)

		self.vertical_reference_line = Line(math_origin, math_vertical_pt2)
		self.horizontal_reference_line = Line(math_origin, math_horizontal_pt)

	def __shift_accessory_coordinate(self, pt):
		math_pt = pt.switch_coordinate_system(self.img) 
		slope_proj, intercept_proj = math_pt.get_projected_line(self.horizontal_reference_line)
		math_intersection = self.vertical_reference_line.calculate_intersection(slope_proj, intercept_proj)
		img_intersection = math_intersection.switch_coordinate_system(self.img)
		return img_intersection

	def __cross_ratio(self, intersection):
		### AC*BD/(CD*AB) = A'C'*B'D'/(C'D'*A'B') ###
		# Image cross ratio
		# AB(scale_length): origin to vertical_pt (scale_pixel_dist)
		# CD: accessory_pt to vertical_pt2
		# BD: vertical_pt to vertical_pt2
		# AC(interested_length): origin to accessory_pt
		AB = self.origin.get_distance(self.vertical_pt.get_point())
		CD = intersection.get_distance(self.vertical_pt2.get_point())
		BD = self.vertical_pt.get_distance(self.vertical_pt2.get_point())
		AC = self.origin.get_distance(intersection.get_point())

		image_ratio = AC*BD/CD/AB

		# World cross ratio
		ABw = self.scale_length
		ADw = self.panel_length
		BDw = self.panel_length - self.scale_length
		
		ACw = image_ratio*ABw*ADw/(BDw+image_ratio*ABw)
		return ACw