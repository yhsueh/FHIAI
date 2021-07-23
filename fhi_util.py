import os
import glob
import sys

from matplotlib import pyplot as plt
from PIL import Image
import cv2

import yolo
import distance_estimator as de
import img_coordinate as ic
import fhi_unet as unet
import geometry as ge


def yolo_detection(img_dir, weights):
	img_dir = os.path.join(os.getcwd(), img_dir)
	yl = yolo.YOLO(model_path=weights)
	yl_results = []
	
	detect_imgs = ['Panel1', 'Panel4', 'Panel5', 'Panel9']
	for img_path in glob.glob(img_dir + r'\*.jpg'):
		basename = os.path.splitext(os.path.basename(img_path))[0]
		if basename not in detect_imgs:
			continue

		print('Processing:', img_path)
		img = Image.open(img_path)
		result = yl.detect_image(img, True)
		result.update({'img_path' : img_path})
		yl_results.append(result)
	return yl_results

def create_masks(un, yl_results):
	def enlarge_roi(roi):
		center = ((roi[0] + roi[2])/2, (roi[1] + roi[3])/2)
		width = 1.4*(roi[2] - roi[0])
		height = 1.4*(roi[3] - roi[1])
		enlarged_roi = (int(center[0]-0.5*width), int(center[1]-0.5*height),
				int(center[0]+0.5*width), int(center[1]+0.5*height))
		return enlarged_roi

	def unet_crop(yl_result):
		img = yl_result['original_img']
		cropped_imgs = []
		masks = []
		mask_coords = []
		for i, roi in enumerate(yl_result['rois']):
			# Enlarge the roi boundry acquired from Yolo
			roi = enlarge_roi(roi)
			
			# Cropped the image
			cropped_img = img[roi[1]:roi[3], roi[0]:roi[2],:]
			mask_coord = (roi[0], roi[1])

			# UNet Detection
			mask = un.detect(cropped_img)

			# Image Processing
			morphology_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
			dilation = cv2.dilate(mask, morphology_kernel, iterations=3)
			mask = cv2.erode(dilation, morphology_kernel, iterations=3)
			_, mask = cv2.threshold(mask, 255/2, 255, cv2.THRESH_BINARY)
			
			# Render Result
			'''
			_, ax = plt.subplots(2)
			ax[0].imshow(cropped_img)
			ax[1].imshow(mask)
			plt.show()
			'''

			#saved_path = os.path.join(os.getcwd(), r'{}.jpg'.format(i))
			cropped_imgs.append(cropped_img)
			masks.append(mask)
			mask_coords.append(mask_coord)

		yl_result.update({'cropped_imgs' : cropped_imgs,
							'masks' : masks,
							'mask_coords' : mask_coords})

	for yl_result in yl_results:
		unet_crop(yl_result)

def unet_detection(yl_results):
	un = unet.UNET()
	un.initialize()
	print('#### unet initialization completed ####')

	create_masks(un, yl_results)
	un.close_session()
	
	print('#### Begin computing real-world distance ####')
	for yl_result in yl_results:
		compute_distance(yl_result)

def compute_distance(yl_result):
	def resize_restoration(mask_itr_pt, cropped_shape):
		unet_resize = 128
		aspect_ratio = cropped_shape[1]/cropped_shape[0] #x/y
		itr_pt = mask_itr_pt.get_point_tuple()

		restored_x = 0
		restored_y = 0

		if aspect_ratio >=1:
			distorted_y = unet_resize / aspect_ratio
			padding_y = (unet_resize - distorted_y)/2

			restored_x = itr_pt[0] * cropped_shape[1] / unet_resize
			restored_y = (itr_pt[1] - padding_y) * cropped_shape[1] / unet_resize
		else:
			distorted_x = unet_resize / aspect_ratio
			padding_x = (unet_resize - distorted_x)/2

			restored_x = (itr_pt[0] - padding_x) * cropped_shape[0] / unet_resize
			restored_y = itr_pt[1] * cropped_shape[0] / unet_resize
		return ge.Point((int(restored_x), int(restored_y)))

	img = yl_result['original_img']
	estimator = de.DistanceEstimator(img)
	estimator.initialize()
	img = estimator.display_reference_pts(img)
	cropped_imgs = yl_result['cropped_imgs']
	masks = yl_result['masks']
	mask_coords = yl_result['mask_coords']
	_, ax = plt.subplots()

	for i, mask in enumerate(masks):
		roi = yl_result['rois'][i]
		class_id = yl_result['class_ids'][i]		
		info = (mask, roi, class_id)
		mask_coord = mask_coords[i]
		cropped_img = cropped_imgs[i]
		pt_itr = None

		if class_id == 0 or class_id == 1:
			accessory = ic.Type1_2Coord(info)
			pt_itr = accessory.get_point_of_interest()
			pt_itr = resize_restoration(pt_itr, cropped_img.shape).add_point(mask_coord)			
			accessory.update_interest_pt(pt_itr.get_point_tuple())
			img = accessory.draw_point_of_interest(img)
		elif class_id == 2 or class_id == 3:
			accessory = ic.Type3_4Coord(info)
			pt_itr = accessory.get_point_of_interest()
			pt_itr = resize_restoration(pt_itr, cropped_img.shape).add_point(mask_coord)			
			accessory.update_interest_pt(pt_itr.get_point_tuple())
			img = accessory.draw_point_of_interest(img)
		else:
			continue

		# Distance estimator
		caption = estimator.estimate(pt_itr)
		ax.text(roi[0], roi[1], caption, color='lime', weight='bold', size=6, backgroundcolor="none")

	print('Process completed')
	img_path = yl_result['img_path']
	img_path = img_path.replace('dataset', r'dataset\distance')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	plt.imshow(img)
	print('Saved path', img_path)
	plt.savefig(img_path, dpi=300)

