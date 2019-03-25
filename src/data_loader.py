import os
import cv2
import random
import scipy.io
import numpy as np
from utils import palette
from PIL import Image

class dataLoader(object):
    def __init__(self, directory, dataset_name, image_height, image_width,\
                 mode='Train', dtype='sbd', num_classes=21):
        self.mode         = mode
        self.dtype        = dtype
        self.image_height = image_height
        self.image_width  = image_width
        self.directory    = directory
        self.num_classes  = num_classes
        self.dataset_name = dataset_name
        self.palette      = palette()

        self.get_data()

    def get_data(self):
        # Load the data from text file
        dataset_path = os.path.join(self.directory, self.dataset_name)
        self.images = [line.rstrip('\n') for line in open(dataset_path)]
        self.max_steps = len(self.images)
        print('Dataset Loaded!')

    def convert_from_color_segmentation(self, arr_3d):
        '''
        Function for converting RGB into 3D Labels --one_hot
        '''
        arr_d = np.zeros((arr_3d.shape[0], arr_3d.shape[1], 21), dtype=np.uint8)
        # slow!
        for i in range(0, arr_3d.shape[0]):
            for j in range(0, arr_3d.shape[1]):
                key = (arr_3d[i,j,0], arr_3d[i,j,1], arr_3d[i,j,2])
                value = self.palette.get(key, 0) # default value if key was not found is 0
                arr_d[i,j,value] = 1

        return arr_d

    def convert_from_value_segmentation(self, arr_2d):
        '''
        Function for converting RGB into 3D Labels --one_hot
        '''
        arr_d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 21), dtype=np.uint8)
        # slow!
        for i in range(0, arr_2d.shape[0]):
            for j in range(0, arr_2d.shape[1]):
                value = arr_2d[i, j]# default value if key was not found is 0
                if value == 255:
                    value = 0
                arr_d[i,j,value] = 1

        return arr_d

    def load_label(self, idx):
        """
        Load label image as height x width integer array of label indices. --SBD
        """
        mat   = scipy.io.loadmat('{}/cls/{}.mat'.format(self.directory, idx))
        label = mat['GTcls'][0]['Segmentation'][0].astype(np.uint8)

        return label

    def randomCrop(self, image, label):
        """
        Crop randomly the image in a sample.
        """
        in_h, in_w, in_c = image.shape
        out_h = self.image_height
        out_w = self.image_width
        height_offset = int(np.random.uniform(0, in_h - out_h + 1))
        width_offset  = int(np.random.uniform(0, in_w - out_w + 1))

        cropped_img = image[height_offset:height_offset + out_h,
        					width_offset:width_offset + out_w, :]

        cropped_lbl = label[height_offset:height_offset + out_h,
        					width_offset:width_offset + out_w, :]

        return cropped_img, cropped_lbl

    def gen_random_data(self):
        while True:
            indices = list(range(len(self.images)))
            random.shuffle(indices)
            for idx in indices:
                yield idx

    def gen_val_data(self):
        while True:
            indices = range(len(self.images))
            for idx in indices:
                yield idx

    def gen_data_batch(self, batch_size):
        # Generate data based on training/validation
        if self.mode == 'Train':
            # Randomize data
            data_gen = self.gen_random_data()
            while True:
                image_batch = []
                label_batch = []
                label_onehot_batch = []
                # Generate training batch
                for _ in range(batch_size):
                    idx = next(data_gen)
                    # Check dataset type
                    if self.dtype == 'sbd':
                        # Get image
                        image_file = os.path.join(self.directory, 'img', str(self.images[idx] + '.jpg'))
                        image = cv2.imread(image_file)
                        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
                        label = self.load_label(idx=self.images[idx])
                        # Resize label
                        label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)
                        label = self.convert_from_value_segmentation(arr_2d=label)
                    else:
                        # Get image
                        image_file = os.path.join(self.directory,  'JPEGImages', str(self.images[idx] + '.jpg'))
                        image = cv2.imread(image_file)
                        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
                        label_file = os.path.join(self.directory,  'SegmentationClass', str(self.images[idx] + '.png'))
                        img_lbl = Image.open(label_file)
                        label = np.array(img_lbl, dtype=np.uint8)
                        # Resize label
                        label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)
                        label = self.convert_from_value_segmentation(arr_2d=label)
                    # Image Augmentation
                    image, label = self.randomCrop(image, label)
                    # Append to generated batch
                    image_batch.append(image)
                    label_batch.append(label)

                yield np.array(image_batch), np.array(label_batch)

        else:
            # Validation Data generation
            val_data_gen = self.gen_val_data()
            while True:
                val_image_batch = []
                val_label_batch = []
                # Generate training batch
                for _ in range(batch_size):
                    idx = next(val_data_gen)
                    # Check dataset type
                    if self.dtype == 'sbd':
                        image_file = os.path.join(self.directory, 'img', str(self.images[idx] + '.jpg'))
                        image = cv2.imread(image_file)
                        image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
                        # Get label
                        label = self.load_label(idx=self.images[idx])
                        # Resize label
                        label = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
                        label = self.convert_from_value_segmentation(arr_2d=label)
                    else:
                        # Get image
                        image_file = os.path.join(self.directory,  'JPEGImages', str(self.images[idx] + '.jpg'))
                        image = cv2.imread(image_file)
                        image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
                        label_file = os.path.join(self.directory,  'SegmentationClass', str(self.images[idx] + '.png'))
                        img_lbl = Image.open(label_file)
                        label = np.array(img_lbl, dtype=np.uint8)
                        # Resize label
                        label = cv2.resize(label, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
                        label = self.convert_from_value_segmentation(arr_2d=label)
                    # Append to generated batch
                    val_image_batch.append(image)
                    val_label_batch.append(label)

                yield np.array(val_image_batch), np.array(val_label_batch)
