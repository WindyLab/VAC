"""
Image concatenat management
"""

import math
import numpy as np
from cfg_vatt import config_train as cfg

class ImageConcatenater():
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        super(ImageConcatenater, self).__init__()
        ## gazebo: 2
        ## real world: pi / 2 (after undistortion)
        ## real world: XXX (without undistortion)
        self.__original_horizontal_fov = cfg['horizontal_fov']  #math.pi / 2   ## rad  
        self.__original_img_width = 640
        self.__original_img_height = 360
        self.__cutted_img_height = 180
        self.__start_id, self.__focal = self.compute_crop_size()

    def compute_crop_size(self):
        """
        h_fov: horizon fov in radian
        img_width: image width in pixel
        Make 90 degree HFOV
        return: crop starting column index, focal_len
        """
        focal = self.__original_img_width / 2 / math.tan(self.__original_horizontal_fov / 2)
        return np.round((self.__original_img_width - 2 * focal) / 2).astype(np.uint8),focal
    
    def get_original_focal(self):
        return self.__original_img_width / 2 / math.tan(self.__original_horizontal_fov / 2)

    def compute_cropped_width(self):
        return self.__original_img_width - 2 * self.__start_id

    def get_original_horizontal_fov(self):
        return self.__original_horizontal_fov
    
    def get_original_img_width(self):
        return self.__original_img_width
    
    def get_cutted_img_height(self):
        return self.__cutted_img_height

    def get_original_img_height(self):
        return self.__original_img_height
    
    def get_cropped_cx(self):
        return self.__original_img_width / 2 - self.__start_id
    
    def get_cropped_cy(self):
        return self.__original_img_height / 2 - (self.__original_img_height - self.__cutted_img_height) ## cut from row 0
       # return self.__original_img_height / 2 - (self.__original_img_height - self.__cutted_img_height)
    
    
    def get_cropped_focal(self):
        return self.__focal

if __name__ == '__main__':
    start_id,focal = ImageConcatenater().compute_crop_size()
    print(start_id,focal)
