import cv2
import glob


class DataGen():
    def __init__(self, image_dir, output_dir) -> None:
        self.image_dir = image_dir
        self.output_dir = output_dir

    def rgb2binary(self, image):
        # read grayscale image
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

        # grayscale to binary
        thresh = 127
        bw_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
        
        cv2.imshow("Binary", bw_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return bw_img
    
    def test(self):
        for image in glob.glob(self.image_dir+"/*.png"):
            self.rgb2binary(image)

image_dir = "test_input"
output_dir = "test_output"
datagen = DataGen(image_dir=image_dir, output_dir=output_dir)
datagen.test()