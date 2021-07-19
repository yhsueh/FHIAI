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

def yolo_detection(img_dir, weights):
	img_dir = os.path.join(os.getcwd(), img_dir)
	yl = yolo.YOLO(model_path=weights)
	yl_results = []
	for img_path in glob.glob(img_dir + r'\*2470.jpg'):
		img = Image.open(img_path)
		result = yl.detect_image(img)

		'''
		img = result['result_img']
		img = cv2.resize(img, (1200,800))
		cv2.imshow('result', img)
		cv2.waitKey()
		sys.exit()
		'''

		yl_results.append(result)

		# Yolo detects only one image
		break
	yl.close_session()
	return yl_results

def create_masks(un, yl_results):
	def enlarge_roi(roi):
		center = ((roi[0] + roi[2])/2, (roi[1] + roi[3])/2)
		width = 1.1*(roi[2] - roi[0])
		height = 1.1*(roi[3] - roi[1])
		enlarged_roi = (int(center[0]-0.5*width), int(center[1]-0.5*height),
				int(center[0]+0.5*width), int(center[1]+0.5*height))
		return enlarged_roi

	def unet_crop(yl_result):
		img = yl_result['original_img']
		masks = []
		mask_coords = []
		for i, roi in enumerate(yl_result['rois']):
			roi = enlarge_roi(roi)
			cropped_img = img[roi[1]:roi[3], roi[0]:roi[2],:]
			mask_coord = (roi[0], roi[1])
			
			nimg = un.detect(cropped_img)
			cv2.imshow('mask', nimg)
			cv2.waitKey()

			#masks.append(un.detect(cropped_img))
			mask_coords.append(mask_coord)

		sys.exit()

		yl_result.update({'masks' : masks, 
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
	img = yl_result['original_img']
	_, ax = plt.subplots(1)

	estimator = de.DistanceEstimator(img)
	estimator.initialize()
	print('#### Distance estimator initialization completed ####')

	img = estimator.display_reference_pts(img)

	masks = yl_result['masks']
	mask_coords = yl_result['mask_coords']

	# Loop through each mask
	print('mask length', len(masks))
	for i, mask in enumerate(masks):
		roi = yl_result['rois'][i]
		class_id = yl_result['class_ids'][i]		
		info = (mask, roi, class_id)

		print(class_id)
		print('mask')
		print(type(mask))
		print(mask.shape)
		cv2.imshow('mask', mask)
		cv2.waitKey()

		mask_coord = mask_coords[i]
		pt_itr = None
		
		if class_id == 0 or class_id == 1:
			print('class 01')
			accessory = ic.Type1_2Coord(info)
			pt_itr = accessory.get_point_of_interest() + mask_coord
			img = accessory.draw_point_of_interest(img)
		elif class_id ==2:
			print('class2')
			accessory = ic.Type3_4Coord(info)
			pt_itr = accessory.get_point_of_interest() + mask_coord
			img = accessory.draw_point_of_interest(img)
		else:
			print('class3')
			accessory = ic.Type3_4Coord(info)
			pt_itr = accessory.get_point_of_interest() + mask_coord
			img = accessory.draw_point_of_interest(img)

		# Distance estimator
		#caption = estimator.estimate(pt_itr)
		#ax.text(roi[1]+10, roi[0] + 30, caption, color='lime', weight='bold', size=6, backgroundcolor="none")

	print('Process completed')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	plt.imshow(img)
	plt.show()


