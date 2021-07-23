import os
import glob
import argparse
import sys

from PIL import Image

import yolo
import fhi_util as fu

def main(img_dir, weights):
    result_yolo = fu.yolo_detection(img_dir, weights)
    fu.unet_detection(result_yolo)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Yolo image detection process')
    parser.add_argument('--weights', required= True,
                        help="Path to weights .h5 file")
    parser.add_argument('--images', required=True,
                        metavar="/path/to/accessory/test_image/",
                        help='Directory of the Accessory test images')
    parser.add_argument('--anchors', required=False)
    parser.add_argument('--classes', required=False)
    args = parser.parse_args()

    if args.weights == 'yolo':
        args.weights = r'weights\yolo\yolov3_608_0706__11_final.h5'
    main(args.images, args.weights)
