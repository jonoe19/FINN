import cv2
import glob
import skimage.exposure
import numpy as np
import sys
from numpy.random import default_rng

class DataGen():
    def __init__(self, image_dir, output_dir, dimension = (512, 512)) -> None:
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.dim = dimension

    def rgb2binary(self, image):
        # read grayscale image
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

        # grayscale to binary
        thresh = 200
        bw_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
        return bw_img
    
    def scaleimg(self, image):
        return cv2.resize(image, self.dim, interpolation= cv2.INTER_LINEAR)

    def addHole(self, image):
        # generate noise
        rng = default_rng(seed=np.random.randint(0, sys.maxsize))
        noise = rng.integers(0, 255, self.dim, np.uint8, True)

        # blur the noise to control size
        blur = cv2.GaussianBlur(noise, (0,0), sigmaX=40, sigmaY=40, borderType = cv2.BORDER_DEFAULT)

        # stretch the blurred image to full dynamic range
        stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0,200)).astype(np.uint8)

        # threshold stretched image to control the size
        thresh = cv2.threshold(stretch, 140, 255, cv2.THRESH_BINARY)[1]

        # apply morphology open and close to smooth out and make 3 channels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.merge([mask,mask,mask])
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        # add mask to input
        result = cv2.add(image, mask)

        return result
        
        
    def test(self):
        for image in glob.glob(self.image_dir+"/*"):
            
            binary_img = self.rgb2binary(image)
            scaled_img = self.scaleimg(binary_img)
            holed_img = self.addHole(scaled_img)
            cv2.imshow("holed_img", holed_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
image_dir = "test_input"
output_dir = "test_output"
datagen = DataGen(image_dir=image_dir, output_dir=output_dir)
datagen.test()