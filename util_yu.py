import numpy as np
import os
from PIL import Image
import sys

OUTPUT_IMAGE_DIR = r'cropped_img'

def crop_image(img, bbox):
	#img.show()
	print(bbox)
	if not os.path.exists(OUTPUT_IMAGE_DIR):
		os.makedirs(OUTPUT_IMAGE_DIR)