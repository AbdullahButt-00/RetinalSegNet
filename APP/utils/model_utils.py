import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K, layers, models
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, GlobalMaxPooling2D,
                                     Reshape, Add, Activation, Multiply, Permute, Lambda,
                                     Concatenate, Conv2D, Conv2DTranspose, BatchNormalization)
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize
from sklearn.metrics import precision_recall_curve, confusion_matrix
import logging

logger = logging.getLogger(__name__)

# ======================== METRICS ========================
def threshold_binarize(x, threshold=0.5):
    ge = tf.greater_equal(x, tf.constant(threshold))
    return tf.where(ge, x=tf.ones_like(x), y=tf.zeros_like(x))

def iou(y_true, y_pred, threshold=0.5):
    y_pred = threshold_binarize(y_pred, threshold)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return intersection / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection)

def dice_coef(y_true, y_pred, threshold=0.5):
    y_pred = threshold_binarize(y_pred, threshold)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    actual_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (actual_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true)*(1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def DiceLoss(y_true, y_pred, smooth=1e-6):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    nominator = 2 * tf.reduce_sum(y_pred * y_true) + smooth
    denominator = tf.reduce_sum(y_pred**2) + tf.reduce_sum(y_true**2) + smooth
    return 1 - nominator / denominator

def bcc_Jaccard_coef_loss(y_true, y_pred):
    return DiceLoss(y_true, y_pred) + (1 - iou(y_true, y_pred, threshold=0.5))

# ======================== CBAM ATTENTION ========================
def cbam_block(cbam_feature, ratio=8):
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]
    
    shared_layer_one = Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1,1,channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    
    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3,1,2))(cbam_feature)
    
    return Multiply()([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        cbam_feature = input_feature
    
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    
    cbam_feature = Conv2D(filters=1, kernel_size=kernel_size, strides=1,
                          padding='same', activation='sigmoid',
                          kernel_initializer='he_normal', use_bias=False)(concat)
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3,1,2))(cbam_feature)
    
    return Multiply()([input_feature, cbam_feature])

# ======================== TRANSFUSE MODEL ========================
def create_transfuse_net(input_shape, key_dim=32):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    conv1 = layers.Conv2D(8, 3, padding='same', activation='relu')(inputs)
    conv1 = layers.MaxPooling2D(2)(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    
    conv2 = layers.Conv2D(16, 3, padding='same', activation='relu')(conv1)
    conv2 = layers.MaxPooling2D(2)(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    
    conv3 = layers.Conv2D(32, 3, padding='same', activation='relu')(conv2)
    conv3 = layers.MaxPooling2D(2)(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    
    # Transformer
    transformer_block = layers.Reshape((-1, conv3.shape[3]))(conv3)
    transformer_block = layers.MultiHeadAttention(num_heads=4, key_dim=key_dim)(transformer_block, transformer_block)
    transformer_block = layers.GlobalAveragePooling1D()(transformer_block)
    transformer_block = layers.Reshape((1,1,transformer_block.shape[1]))(transformer_block)
    transformer_block = layers.Lambda(lambda x: tf.tile(x, [1, conv3.shape[1], conv3.shape[2], 1]))(transformer_block)
    
    att1 = cbam_block(transformer_block)
    fused_features = layers.Concatenate()([conv3, att1])
    
    # Decoder
    dec1 = layers.Conv2DTranspose(32,3,strides=2,padding='same',activation='relu')(fused_features)
    att2 = cbam_block(dec1)
    dec1 = layers.Concatenate()([att2, conv2])
    dec1 = layers.Conv2D(32,3,padding='same',activation='relu')(dec1)
    
    dec2 = layers.Conv2DTranspose(16,3,strides=2,padding='same',activation='relu')(dec1)
    att3 = cbam_block(dec2)
    dec2 = layers.Concatenate()([att3, conv1])
    dec2 = layers.Conv2D(16,3,padding='same',activation='relu')(dec2)
    
    dec3 = layers.Conv2DTranspose(8,3,strides=2,padding='same',activation='relu')(dec2)
    dec3 = layers.Conv2D(8,3,padding='same',activation='relu')(dec3)
    
    output_BV = layers.Conv2D(1,1,activation='sigmoid',name='final_output1')(dec3)
    output_OD = layers.Conv2D(1,1,activation='sigmoid',name='final_output2')(dec3)
    
    model = models.Model(inputs=inputs, outputs=[output_BV, output_OD])
    return model

# ======================== DATA LOADING ========================
def load_image(image_path, im_height=512, im_width=512):
    """Load and preprocess a single image"""
    try:
        img = img_to_array(load_img(image_path, color_mode='rgb'))
        img_resized = resize(img, (im_height, im_width, 3), preserve_range=True) / 255.0
        return np.array([img_resized]).astype(np.float32)
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None

def get_data(path, im_height=512, im_width=512):
    """Load dataset from folder structure"""
    images_path = os.path.join(path, 'images')
    masks_path = os.path.join(path, 'mask')
    
    image_files = sorted([f for f in os.listdir(images_path) if f.endswith(('.tif','.png','.jpg'))])
    n = len(image_files)
    
    X = np.zeros((n, im_height, im_width, 3), dtype=np.float32)
    y = np.zeros((n, im_height, im_width, 1), dtype=np.float32)
    
    for i, img_name in enumerate(image_files):
        img = img_to_array(load_img(os.path.join(images_path, img_name), color_mode='rgb'))
        X[i] = resize(img, (im_height, im_width, 3), preserve_range=True) / 255.0
        
        mask_name = img_name.replace('.tif','_mask.gif').replace('.png','_mask.gif').replace('.jpg','_mask.gif')
        mask_path = os.path.join(masks_path, mask_name)
        if os.path.exists(mask_path):
            mask = img_to_array(load_img(mask_path, color_mode='grayscale'))
            y[i] = resize(mask, (im_height, im_width, 1), preserve_range=True) / 255.0
    
    return X, y