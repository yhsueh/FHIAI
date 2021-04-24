import yolo
import os
import glob
from PIL import Image

dirname = os.path.dirname(__file__)


def detect_image(path):
    image = Image.open(path)
    yolo_obj = yolo.YOLO()
    r_image = yolo_obj.detect_image(image=image)
    save_path = os.path.join(dirname, 'output_image/' + path[-5:])
    r_image.save(save_path)


def main():
    for path in glob.glob('test_image/*.jpg'):
        image_path = os.path.join(dirname, path)
        detect_image(image_path)


if __name__ == '__main__':
    main()
