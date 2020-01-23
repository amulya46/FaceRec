'''
@Author: David Vu
Run the pretrained model to extract 128D face features
'''
import tensorflow.compat.v1 as tf
from architecture import inception_resnet_v1 as resnet
from tensorflow.python.platform import gfile
import numpy as np
import os 
from edgetpu.basic.basic_engine import BasicEngine
#import numpy as np
#from PIL import Image 
#input = "./images/p1.jpg"
#model_path = 'open_pose_tflite'
#target_size=(432, 368)
#output_size = [54,46,57]
#'''load the image'''
#image = Image.open(image_path)
#image = image.resize(target_size, Image.ANTIALIAS)
#image = np.array(image).flatten()
#
#'''load the model'''
#engine = BasicEngine(model_path)
#result = engine.RunInference(input_tensor = image)
#process_time = result[0]
#my_model_output = result[1]. reshape(output_size)

class FaceFeature(object):
    def __init__(self, face_rec_graph, model_path = 'models/converted_model_new_512.tflite'):

        '''
        :param face_rec_sess: FaceRecSession object
        :param model_path:
        '''
        print("Loading model...")
        with face_rec_graph.graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                self.__load_model(model_path)
                self.x = tf.get_default_graph() \
                                            .get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph() \
                                    .get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph() \
                                                     .get_tensor_by_name("phase_train:0")                    

                print("Model loaded")

    def __load_model(self, model_path):
        print("Loading model...")
        BasicEngine(model_path)
        print("Model loaded")

    # for input images
    def get_features(self, input_imgs):
        images = load_data_list(input_imgs,160)
        feed_dict = {self.x: images, self.phase_train_placeholder: False}
        return self.sess.run(self.embeddings, feed_dict = feed_dict)


def tensorization(img):
    '''
    Prepare the imgs before input into model
    :param img: Single face image
    :return tensor: numpy array in shape(n, 160, 160, 3) ready for input to cnn
    '''
    tensor = img.reshape(-1, Config.Align.IMAGE_SIZE, Config.Align.IMAGE_SIZE, 3)
    return tensor

#some image preprocess stuff
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y

def load_data_list(imgList, image_size, do_prewhiten=True):
    images = np.zeros((len(imgList), image_size, image_size, 3))
    i = 0
    for img in imgList:
        if img is not None:
            if do_prewhiten:
                img = prewhiten(img)
            images[i, :, :, :] = img
            i += 1
    return images