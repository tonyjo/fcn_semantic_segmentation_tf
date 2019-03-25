import os
import cv2
import random
import numpy as np
from tqdm import tqdm

class dataLoader(object):
    def __init__(self, directory, dataset_name, image_height, image_width,\
                 mode='Train', dtype='sbd', num_classes=40):
        self.mode         = mode
        self.dtype        = dtype
        self.image_height = image_height
        self.image_width  = image_width
        self.directory    = directory
        self.num_classes  = num_classes
        self.dataset_name = dataset_name
        self.get_data()

    def get_data(self):
        # Load the data from text file
        self.images = [line.rstrip('\n') for line in open(self.dataset_name)]
        self.max_steps = len(self.images)
        print('Dataset Loaded!')

    def convert_from_color_to_labels(self, arr_d):
        '''
         Function for converting RGB into 2D Labels-- VOC
        '''
        arr_2d  = np.zeros((arr_d.shape[0], arr_d.shape[1]))
        # slow!
        for i in range(0, arr_d.shape[0]):
            for j in range(0, arr_d.shape[1]):
                key   = (arr_d[i,j,0], arr_d[i,j,1], arr_d[i,j,2])
                value = palette.get(key, 0) # default value if key was not found is 0
                arr_2d[i,j] = value # default value if key was not found is 0

        return arr_2d

    def load_label(self, idx):
        """
        Load label image as height x width integer array of label indices. --SBD
        """
        mat   = scipy.io.loadmat('{}/cls/{}.mat'.format(self.dataset_dir, idx))
        label = mat['GTcls'][0]['Segmentation'][0].astype(np.uint8)

        return label

    def centeredCrop(self, img, output_side_length=224):
        """
        Center Crop the image in a sample.
        """
        height, width, depth = img.shape
        new_height = output_side_length
        new_width = output_side_length

        if height > width:
            new_height = output_side_length * height / width
        else:
            new_width = output_side_length * width / height

        height_offset = (new_height - output_side_length) / 2
        width_offset  = (new_width - output_side_length) / 2
        cropped_img   = img[height_offset:height_offset + output_side_length,
                            width_offset:width_offset + output_side_length]

        return cropped_img

    def randomCrop(self, image):
        """
        Crop randomly the image in a sample.
        """
        in_h, in_w, in_c = image.shape
        out_h = self.image_height
        out_w = self.image_width
        height_offset = int(np.random.uniform(0, in_h - out_h + 1))
        width_offset  = int(np.random.uniform(0, in_w - out_w + 1))

        cropped_img = image[height_offset:height_offset + out_h,
        					width_offset:width_offset + out_w]

        return cropped_img

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
                    image_file = os.path.join(self.dataset_dir, str(self.images[index] + '.jpg'))
                    image = cv2.imread(image)
                    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
                    # Subtract image mean
                    if self.dtype == 'sbd':
                        image = self.preprocess_image(image)
                    else:
                        image = image/127.5 - 1.0
                    # Image Augmentation
                    image = self.randomCrop(image)
                    image = self.randomFlip(image)
                    #image = self.blurGaussian(image, sigma=0.5)
                    image = self.channelShuffle(image)
                    # Append to generated batch
                    image_batch.append(image)
                    label_batch.append(label)
                    label_onehot_batch.append(label_onehot)

                yield np.array(image_batch), np.array(label_batch), np.array(label_onehot_batch)

        else:
            # Validation Data generation
            val_data_gen = self.gen_val_data()
            while True:
                val_image_batch  = []
                val_label_batch  = []
                val_label_onehot_batch = []
                # Generate training batch
                for _ in range(batch_size):
                    image = next(val_data_gen)
                    image = cv2.imread(image)
                    image = cv2.resize(image, (384, 256))
                    # Subtract image mean
                    if self.mean_type == 'image_mean':
                        label = self.load_label(idx=image)
                    else:
                        image = image/127.5 - 1.0
                    # Image Augmentation
                    #image = self.centeredCrop(image)
                    image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
                    # Append to generated batch
                    val_image_batch.append(image)
                    val_label_batch.append(label)
                    val_label_onehot_batch.append(label_onehot)

                yield np.array(val_image_batch), np.array(val_label_batch), np.array(val_label_onehot_batch)
