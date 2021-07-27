import os
import sys

import cv2
import numpy as np
import tensorflow as tf

import unet_utils as utils

class UNET():
    def __init__(self, model_dir):
        model_dir = os.path.join(os.getcwd(), model_dir)
        self.model_path =os.path.join(model_dir, 'trained_model.ckpt')

    def initialize(self):
        self.sess = tf.Session()  # tf.Session()
        meta_path = self.model_path + '.meta'
        model_loader = tf.train.import_meta_graph(meta_path)
        model_loader.restore(self.sess, self.model_path)
        graph = tf.get_default_graph()

        self.tf_x = graph.get_tensor_by_name('tf_x:0')
        self.tf_predict = graph.get_tensor_by_name('predict:0')

        self.is_training = graph.get_tensor_by_name('is_training:0')
        '''
        try:
            self.is_training = graph.get_tensor_by_name('is_training_1:0')
        except KeyError as e:
            print("[warn] %s -> is_training_1 not found. Try to Find is_training ..." % e)
        '''

    def detect(self, img):
        height, width = self.tf_x.get_shape().as_list()[1:3]
        image = utils.cv_letterbox(img, height, width)
        image_in = np.expand_dims(np.float32(image) / 255 - 0.5, axis=0)
        predict_mask = self.sess.run(self.tf_predict, feed_dict={self.tf_x: image_in, self.is_training: False})
        predict_mask = np.uint8(np.clip(predict_mask[0, ..., 0] + 0.5, 0., 1.) * 255)
        #predict_mask = cv2.cvtColor(predict_mask, cv2.COLOR_GRAY2BGR)
        return predict_mask

    def close_session(self):
        self.sess.close()

if __name__ == '__main__':
    img_path = os.path.join(os.getcwd(), r'test_data\IMG_1222_0_1.jpg')
    img = cv2.imread(img_path)

    un = UNET()
    un.initialize()
    un.detect(img)