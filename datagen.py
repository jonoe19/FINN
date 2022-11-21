import cv2
import glob
import skimage.exposure
import numpy as np
import sys
from numpy.random import default_rng

class DataGen():
    def __init__(self, image_dir, output_dir, dimension = (256, 256)) -> None:
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.dim = dimension


    def rgb2binary(self, image):
        # grayscale to binary
        thresh = 200
        bw_img = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]
        return bw_img
    
    def scaleimg(self, image):
        return cv2.resize(image, self.dim, interpolation= cv2.INTER_LINEAR)

    def addHole(self, image):
        # generate noise
        rng = default_rng(seed=np.random.randint(0, sys.maxsize))
        noise = rng.integers(0, 255, self.dim, np.uint8, True)

        # blur the noise to control size
        blur = cv2.GaussianBlur(noise, (0,0), sigmaX=30, sigmaY=30, borderType = cv2.BORDER_DEFAULT)

        # stretch the blurred image to full dynamic range
        stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0,200)).astype(np.uint8)

        # threshold stretched image to control the size
        thresh = cv2.threshold(stretch, 130, 255, cv2.THRESH_BINARY)[1]

        # apply morphology open and close to smooth out and make 3 channels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.merge([mask,mask,mask])
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        # add mask to input
        result = cv2.add(image, mask)

        return result
        
    def augmentFence(self, image):
        dim = image.shape
        # flipping
        i = np.random.randint(-1, 2)
        if i != 2:
            image = cv2.flip(image, i) 

        # rotating
        j = np.random.randint(0,3)
        for i in range(0,j): 
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        
        # warping
        pix = 5
        print(dim)
        pt_A = [np.random.randint(0,pix),np.random.randint(0,pix)]
        pt_B = [dim[0]-np.random.randint(0,pix),np.random.randint(0,pix)]
        pt_C = [np.random.randint(0,pix),dim[1]-np.random.randint(0,pix)]
        pt_D = [dim[0]-np.random.randint(0,pix),dim[1]-np.random.randint(0,pix)]
        inputpoints = np.float32([pt_A, pt_B, pt_C, pt_D])
        print('in',inputpoints)
        outputpoints = np.float32([[0,0],[dim[0],0],[0,dim[1]], [dim[0],dim[1]]])
        print('out', outputpoints)

        M = cv2.getPerspectiveTransform(inputpoints,outputpoints)
        image = cv2.warpPerspective(image,M,(dim[1], dim[0]),flags=cv2.INTER_LINEAR)
        return image
    
    def test(self, num_images):
        i = 0
        while (i <= num_images):
            for rural in glob.glob(self.image_dir+"/rural"+"/*"):
                rural_gray = cv2.imread(rural, cv2.IMREAD_GRAYSCALE)
                rural_scaled_img = self.scaleimg(rural_gray)
                for fence in glob.glob(self.image_dir+"/fence"+"/*"):
                    # fence
                    fence_gray = cv2.imread(fence, cv2.IMREAD_GRAYSCALE)
                    fence_binary_img = self.rgb2binary(fence_gray)
                    fence_augmented_img = self.augmentFence(fence_binary_img)
                    fence_scaled_img = self.scaleimg(fence_augmented_img)
                    fence_holed_img = self.addHole(fence_scaled_img)

                    # rural
                    rows, cols = rural_scaled_img.shape
                    rural_scaled_img = rural_scaled_img[0:rows, 0:cols]
                    mask = cv2.bitwise_or(rural_scaled_img, rural_scaled_img, mask=fence_holed_img)
                    result = cv2.add(rural_scaled_img, mask)
                    result = cv2.addWeighted(rural_scaled_img, 0.4, mask, 1.0, 0)
                    
                    i += 1
                    if (i > num_images):
                        return
                    
                    cv2.imshow("holed_img", fence_holed_img)
                    cv2.imshow("rural_scaled_img", rural_scaled_img)
                    cv2.imshow("result", result)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            
image_dir = "raw_images"
output_dir = "test_outpt"
datagen = DataGen(image_dir=image_dir, output_dir=output_dir)
datagen.test(5)

