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
    s   = pred.shape
    for i in range(pred.shape[0]):
        img = np.ones((s[1], s[2], 3))

        for h in range(s[1]):
            for w in range(s[1]):
                ui = u[h,w]
                vi = v[h,w]

                img[ui, vi, :] = 255.
        out.append(img)

    return np.float32(np.uint8(out))

def colorize(value, name='pred_to_image'):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    """

    with tf.variable_scope(name), tf.device('/cpu:0'):
        img = tf.py_func(flow_to_image, [value], tf.float32, stateful=False)
        img.set_shape(value.get_shape().as_list()[0:-1]+[3])
        value = value / 127.5 - 1.

        return img
