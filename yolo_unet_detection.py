import os
import glob
import argparse
import sys
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from PIL import Image
import json

import yolo
import fhi_util as fu

def main(input_json):
    verify_paths(input_json)
    result_yolo = fu.yolo_detection(input_json)
    fu.unet_detection(input_json, result_yolo)

def verify_paths(input_json):
    cwd = os.getcwd()
    try:
        os.path.isdir(os.path.join(cwd, input_json['image_dir']))
        os.path.isdir(os.path.join(cwd, input_json['unet_weight_dir']))
        os.path.isdir(os.path.join(cwd, input_json['yolo_output_dir']))
        os.path.isdir(os.path.join(cwd, input_json['unet_put_dir']))
        os.path.isfile(os.path.join(cwd, input_json['yolo_weights']))
    except:
        print('A path does not exist')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Yolo image detection process')
    parser.add_argument('--user_input', required= True,
                        help="Path to weights .h5 file")
    args = parser.parse_args()

    with open(args.user_input, 'r') as f:
        input_json = json.load(f)

    main(input_json)
