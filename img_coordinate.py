import sys
import os

import cv2
import numpy as np

from geometry import Point

class ImgCoord():
	def __init__(self, info):
		self.mask = info[0].astype(np.uint8)
		self.roi = info[1]
		self.class_id= info[2]

	def draw_point_of_interest(self, img):
		img = cv2.circle(img, (self.x_interest, self.y_interest), 0, (0, 0, 255), 5)
		img = cv2.circle(img, (self.x_interest, self.y_interest), 15, (100,255,100), 3)
		return img

	def get_point_of_interest(self):
		raise NotImplementedError()

class Type1_2Coord(ImgCoord):
	def __init__(self, info):
		super().__init__(info)

	def get_point_of_interest(self):
		epsilon = 2
		contours, _ = cv2.findContours(image=self.mask,
									   mode=cv2.RETR_EXTERNAL,
									   method=cv2.CHAIN_APPROX_NONE)
		contour = contours[0][:,0,:]
		pts_x = contour[:,0]
		pts_y = contour[:,1]
		pts_ux = np.mean(pts_x)

		### Select the points near x mean ###
		selected_pts_mask = (pts_x < pts_ux + epsilon) & (pts_x > pts_ux - epsilon)
		selected_pts_x = pts_x[selected_pts_mask]
		selected_pts_y = pts_y[selected_pts_mask]
		selected_pts_uy = np.mean(selected_pts_y)

		### Find min y that is also greater than y_mean ###
		conditioned_min_y = 99999
		for i, y in enumerate(selected_pts_y):
			if y < conditioned_min_y and y > selected_pts_uy:
				conditioned_min_y = y

		### Take the average of x coordinates of the points with same y coordinates ###
		selected_pts_y_mask = selected_pts_y == conditioned_min_y
		interested_pts_x = selected_pts_x[selected_pts_y_mask]
		self.x_interest = int(np.mean(interested_pts_x))
		self.y_interest = conditioned_min_y
		return Point((self.x_interest,self.y_interest))

class Type3_4Coord(ImgCoord):
	def __init__(self, info):
		super().__init__(info)

	def get_point_of_interest(self):
		approx_y_selection_range = 20
		contours, _ = cv2.findContours(image=self.mask,
									   mode=cv2.RETR_EXTERNAL,
									   method=cv2.CHAIN_APPROX_NONE)
		approx = cv2.approxPolyDP(contours[0], 20, True)
		approx = approx[:,0,:]
		approx_y = approx[:,1]
		approx_y_max = np.max(approx_y)
		selected_pt_mask_max = approx_y > (approx_y_max-approx_y_selection_range)
		approx_max_pts = approx[selected_pt_mask_max]

		approx_left_corner = approx_max_pts[0]
		for pt in approx_max_pts:
			if pt[0] < approx_left_corner[0]:
				approx_left_corner = pt
		self.x_interest = approx_left_corner[0]
		self.y_interest = approx_left_corner[1]
		return Point(approx_left_corner)