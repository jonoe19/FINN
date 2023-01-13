import cv2
import glob
import skimage.exposure
import numpy as np
import sys
from numpy.random import default_rng
from tqdm import tqdm


class DataGen():
    '''
    Init
    '''

    def __init__(self, image_dir, output_dir, dimension=(256, 256)) -> None:
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.dim = dimension

    '''
    Given an RGB image, returns a binary representation of the image.
    Pixels are split at a luminocity of 200.
    '''

    def rgb2binary(self, image):
        # grayscale to binary
        thresh = 200
        bw_img = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]
        return bw_img

    '''
    Scales the given image to the size provided in the dimension attribute
    '''

    def scaleimg(self, image):
        return cv2.resize(image, self.dim, interpolation=cv2.INTER_LINEAR)

    '''
    Generates a random 'hole' mask and overlays it on the provided image.
    '''

    def addHole(self, image):
        # For
        pixels = False
        while (not pixels):
            # generate noise
            rng = default_rng(seed=np.random.randint(0, sys.maxsize))
            noise = rng.integers(0, 255, self.dim, np.uint8, True)

            # blur the noise to control size
            blur = cv2.GaussianBlur(
                noise, (0, 0), sigmaX=30, sigmaY=30, borderType=cv2.BORDER_DEFAULT)

            # stretch the blurred image to full dynamic range
            stretch = skimage.exposure.rescale_intensity(
                blur, in_range='image', out_range=(0, 200)).astype(np.uint8)

            # threshold stretched image to control the size
            thresh = cv2.threshold(stretch, 130, 255, cv2.THRESH_BINARY)[1]

            # apply morphology open and close to smooth out and make 3 channels
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.merge([mask, mask, mask])
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

            pixels = self.check4Hole(mask)

        # add mask to input
        result = cv2.add(image, mask)

        return result

    def check4Hole(self, mask) -> bool:
        mask_pixels = mask.sum()/255
        if mask_pixels > 500 and mask_pixels < 30000:
            return True
        return False

    '''
    Generates new fence patterns by applying mulitple rotations
    '''

    def augmentFence(self, image):
        augmented_images = []
        for i in range(-1, 3):
            augmented_image = image
            if i != 2:
                augmented_image = cv2.flip(image, i)
            for j in range(0, 4):
                final_image = augmented_image
                for i in range(0, j):
                    final_image = cv2.rotate(
                        final_image, cv2.ROTATE_90_CLOCKWISE)
                augmented_images.append(final_image)
        return augmented_images

    '''
    Returns a list containing the background image and a mirrored version of it
    '''

    def augmentBG(self, image):
        return [image, cv2.flip(image, 1)]

    '''
    Iterates over rural images applying data augmentation to obtain an adequate dataset for the training process.
    '''

    def image_gen(self) -> None:
        total_tqdm = 35*2*8*16*2
        with tqdm(total=total_tqdm) as pbar:
            total_rural_index = 0
            for rural in glob.glob(self.image_dir+"/rural"+"/*"):
                rural_gray = cv2.imread(rural, cv2.IMREAD_GRAYSCALE)
                rural_scaled_img = self.scaleimg(rural_gray)
                rural_augmentation_list = self.augmentBG(rural_scaled_img)
                total_fence_index = 0
                if total_rural_index < 29:
                    dir = f'{self.output_dir}/train'
                else:
                    dir = f'{self.output_dir}/test'

                for fence in glob.glob(self.image_dir+"/fence"+"/*"):
                    # fence
                    fence_gray = cv2.imread(fence, cv2.IMREAD_GRAYSCALE)
                    fence_binary_img = self.rgb2binary(fence_gray)
                    fence_scaled_img = self.scaleimg(fence_binary_img)
                    fence_augmented_img_list = self.augmentFence(
                        fence_scaled_img)
                    fence_augmentation_list_hole = []
                    fence_augmentation_list_nohole = []
                    for img in fence_augmented_img_list:
                        fence_augmentation_list_hole.append(self.addHole(img))
                        fence_augmentation_list_nohole.append(img)

                    # rural
                    rural_index = 0  # For naming the output file
                    for rural_img in rural_augmentation_list:
                        rows, cols = rural_img.shape
                        rural_img = rural_img[0:rows, 0:cols]

                        fence_index = 0
                        for fence_img in fence_augmentation_list_hole:

                            mask = cv2.bitwise_or(
                                rural_scaled_img, rural_scaled_img, mask=fence_img)
                            result = cv2.add(rural_img, mask)
                            result = cv2.addWeighted(
                                rural_img, 0.4, mask, 1.0, 0)
                            cv2.imwrite(
                                f'{dir}/hole/R{total_rural_index}r{rural_index}F{total_fence_index}f{fence_index}hole.png', result)
                            fence_index += 1
                            pbar.update(1)

                        fence_index = 0
                        for fence_img in fence_augmentation_list_nohole:
                            mask = cv2.bitwise_or(
                                rural_scaled_img, rural_scaled_img, mask=fence_img)
                            result = cv2.add(rural_img, mask)
                            result = cv2.addWeighted(
                                rural_img, 0.4, mask, 1.0, 0)
                            cv2.imwrite(
                                f'{dir}/nohole/R{total_rural_index}r{rural_index}F{total_fence_index}f{fence_index}nohole.png', result)
                            fence_index += 1
                            pbar.update(1)

                        rural_index += 1
                    total_fence_index += 1
                total_rural_index += 1


image_dir = "raw_images"
output_dir = "data"
datagen = DataGen(image_dir=image_dir, output_dir=output_dir)
datagen.image_gen()
