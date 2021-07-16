import yolo
import os
import glob
from PIL import Image
import argparse
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(root_dir, r'cropped_img\input_img')

def detection(img_path):
    image = Image.open(img_path)
    yolo_obj = yolo.YOLO()
    img, bboxes = yolo_obj.detect_image(image=image, draw_flag=False)
    yolo_obj.close_session()
    cropped_img(img, img_path, bboxes)

def main():
    for img_path in glob.glob(img_dir + r'\*.jpg'):
        detection(img_path)

if __name__ == '__main__':
    parser = argparse.ArgumentPasrer(description='Yolo image detection process')
    parser.add_argument('--weights', required= True,
                        help="Path to weights .h5 file")
    parser.add_argument('--test_image', required=True,
                        metavar="/path/to/accessory/test_image/",
                        help='Directory of the Accessory test images')
    args = parser.parse_args()

    arg.weights
    main()
