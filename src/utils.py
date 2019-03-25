import numpy as np
import matplotlib
import matplotlib.cm
import tensorflow as tf

# Palette
def palette():
    '''
    Function for Selecting Labels for RGB values for PASCAL-VOC dataset
    '''
    palette = { (  0,   0,   0) : 0 ,
                (128,   0,   0) : 1 ,
                (  0, 128,   0) : 2 ,
                (128, 128,   0) : 3 ,
                (  0,   0, 128) : 4 ,
                (128,   0, 128) : 5 ,
                (  0, 128, 128) : 6 ,
                (128, 128, 128) : 7 ,
                ( 64,   0,   0) : 8 ,
                (192,   0,   0) : 9 ,
                ( 64, 128,   0) : 10,
                (192, 128,   0) : 11,
                ( 64,   0, 128) : 12,
                (192,   0, 128) : 13,
                ( 64, 128, 128) : 14,
                (192, 128, 128) : 15,
                (  0,  64,   0) : 16,
                (128,  64,   0) : 17,
                (  0, 192,   0) : 18,
                (128, 192,   0) : 19,
                (  0,  64, 128) : 20 }

    return palette

def rev_pascal_palette():
    palette = { 0: (  0,   0,   0),
                1: (128,   0,   0),
                2: (  0, 128,   0),
                3: (128, 128,   0),
                4: (  0,   0, 128),
                5: (128,   0, 128),
                6: (  0, 128, 128),
                7: (128, 128, 128),
                8: ( 64,   0,   0),
                9: (192,   0,   0),
                10: ( 64, 128,   0),
                11: (192, 128,   0),
                12: ( 64,   0, 128),
                13: (192,   0, 128),
                14: ( 64, 128, 128),
                15: (192, 128, 128),
                16: (  0,  64,   0),
                17: (128,  64,   0),
                18: (  0, 192,   0),
                19: (128, 192,   0),
                20: (  0,  64, 128)}

    return palette

def convert_from_segmentation_color(arr_2d):
    '''
    Function for converting Labels into RGB
    '''
    arr_3d  = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    palette = rev_pascal_palette()
    # slow!
    for i in range(0, arr_2d.shape[0]):
        for j in range(0, arr_2d.shape[1]):
            key = arr_2d[i,j]
            arr_3d[i,j,:] = palette.get(key, 0) # default value if key was not found is 0

    return arr_3d

def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# Bilinear interpolation kernel weights
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """
    Assigns 2D bilinear kernel weights to ConvTranspose2D layer.
    """
    filt = upsample_filt(size=kernel_size)

    weight = np.zeros((kernel_size, kernel_size, in_channels, out_channels), dtype=np.float64)

    for i in range(in_channels):
        weight[:, :, i, i] = filt

    return weight

def colorize_semantic_seg(pred):
    """
    Convert prediction labels to color image using PASCAL-VOC pallete.
    """
    out = []
    for i in range(pred.shape[0]):
        img = convert_from_segmentation_color(arr_2d=pred[i, :, :])

        out.append(img)

    return np.float32(np.uint8(out))

def colorize(value, name='pred_to_image'):
    """
    A utility function for TensorFlow that maps prediction to segmentation color map
    """
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img = tf.py_func(colorize_semantic_seg, [value], tf.float32, stateful=False)
        img.set_shape(value.get_shape().as_list() + [3])
        print(img.get_shape())
        return img
