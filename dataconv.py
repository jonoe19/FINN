import cv2
import glob
import skimage.exposure
import numpy as np
import sys
import os
from numpy.random import default_rng
from tqdm import tqdm

class DataConv():
    '''
    Init
    '''
    def __init__(self, image_dir, output_dir, dimension = (256, 256)) -> None:
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.dim = dimension
    
    '''
    Scales the given image to the size provided in the dimension attribute
    '''
    def scaleimg(self, image):
        return cv2.resize(image, self.dim, interpolation= cv2.INTER_LINEAR)

    def convertimg(self, hole):
        for image in glob.glob(self.image_dir+hole+"/*"):
            img_name = os.path.basename(image)
            image_gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            scaled_image = self.scaleimg(image_gray)
            cv2.imwrite(f'{self.output_dir}/{hole}/{img_name}', scaled_image)

    '''
    Iterates over rural images applying data augmentation to obtain an adequate dataset for the training process.
    '''
    def image_gen(self) -> None:
        self.convertimg(hole="/hole")
        self.convertimg(hole="/nohole")
                           
image_dir = "raw_images/real"
output_dir = "data/real"
datagen = DataConv(image_dir=image_dir, output_dir=output_dir)
datagen.image_gen()

