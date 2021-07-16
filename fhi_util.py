import yolo
import os
import glob
import fhi_unet as unet
from PIL import Image

def yolo_detection(img_dir, weights):
	img_dir = os.path.join(os.getcwd(), img_dir)
	yl = yolo.YOLO(model_path=weights)
	yl_results = []
	for img_path in glob.glob(img_dir + r'\*.jpg'):
		img = Image.open(img_path)
		yl_results.append(yl.detect_image(img))
	yl.close_session()
	return yl_results

def yolo_unet_converter(y1_results):
	def enlarge_rois(rois):
		enlarged_rois = []
		for roi in rois:
			center = ((roi[0] + roi[2])/2, (roi[1] + roi[3])/2)
			width = 1.1*(roi[2] - roi[1])
			height = 1.1*(roi[3] - roi[0])
			enlarged_roi.append((center[0]-0.5*width, center[1]-0.5*height,
				center[0]+0.5*width, center[1]+0.5*height))
		return enlarged_rois
	
	for y1_result in y1_results:
		cropped_imgs = []
		img = y1_result['result_img']
		rois = y1_result['rois']
		class_ids = y1_result['class_ids']

		enlarged_rois = enlarge_rois(rois)
		# Pass cropped images to Img_coordinate class
		# Save the the coordinates of points of interest
		# Return the results for distance estimator

def unet_detection(y1_results):
	un = unet.UNET()
	un.initialize()
	# yolo_unet_converter


