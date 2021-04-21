import yolo
import os
from PIL import Image

dirname = os.path.dirname(__file__)
image_name = 'File_007.jpeg'
image_path = os.path.join(dirname, 'test_image/' + image_name)

image = Image.open(image_path)
yolo = yolo.YOLO()
r_image = yolo.detect_image(image=image)
save_path = os.path.join(dirname, 'output_image/' + image_name)
r_image.save(save_path)
