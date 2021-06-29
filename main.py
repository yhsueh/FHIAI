import yolo
import os
import glob
from PIL import Image
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(root_dir, r'cropped_img\input_img')
cropped_img_dir = os.path.join(root_dir, r'cropped_img\output_img\image')

def cropped_img(img, img_path, bboxes):
    origin_img = img
    origin_img_name = os.path.splitext(os.path.basename(img_path))

    for i, bbox in enumerate(bboxes):
        cropped_img = origin_img.crop(bbox)

        new_file_name = '{}_{}{}'.format(origin_img_name[0], i, origin_img_name[1])
        save_path = os.path.join(cropped_img_dir, new_file_name)
        cropped_img.save(save_path)
        print('successfully saved: {}'.format(save_path))

def detect_cropped_image(img_path):
    image = Image.open(img_path)
    yolo_obj = yolo.YOLO()
    img, bboxes = yolo_obj.detect_image(image=image, draw_flag=False)
    cropped_img(img, img_path, bboxes)

def detect_cropped_image(img_path):
    image = Image.open(img_path)
    yolo_obj = yolo.YOLO()
    img, bboxes = yolo_obj.detect_image(image=image)
    img.show()

def main():
    for img_path in glob.glob(img_dir + r'\*.jpg'):
        detect_cropped_image(img_path)

if __name__ == '__main__':
    main()
