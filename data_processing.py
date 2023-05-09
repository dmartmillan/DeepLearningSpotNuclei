import numpy as np
import tensorflow as tf
import random
from IPython.core.display import Image, display
import keras.backend as K

seed = 2023
np.random.seed(seed) 

class data_grinder:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.isImageGrayscale = lambda img: all(map(lambda p: p[0] == p[1] == p[2], img.getdata()))
                                   
    def preprocessImages(self, contrast_ratio = 1.5, threshold = 0.6):
        images = self.data_dict['images']
        preproc = []
        for image in images:
            
            # Load Image
            grayscale = self.isImageGrayscale(image)
            # Convert to grayscale
            image = tf.image.rgb_to_grayscale(image)
            
            # Invert Image if original was colored or background treshold is reached
            if not grayscale or np.sum(image/255) > threshold*image.shape[0]*image.shape[1]:
                image = 255 - image
            
            # Normalize
            image = image - np.min(image)
            if np.max(image) > 0:
                image = np.round(image * (255 / np.max(image))).astype(np.uint8)
                
                # Increase Contrast
                image = tf.image.adjust_contrast(image, contrast_ratio)

            preproc.append(image)
        self.data_dict['preprocessed_images'] = preproc

    def croporresizeImages(self, hasmask = True, window_shape = (128, 128), method = 'crop'):
        preproc = self.data_dict['preprocessed_images']
        if hasmask: masks = self.data_dict['masks']

        X = []
        Y = []
        idx = []
        sizes = []
        j = 0
        for i, image in enumerate(preproc):
            image_size = image.shape
            sizes.append(image_size[:2])
            if method == 'resize':
                image = tf.image.resize(image, window_shape, method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                X.append(image)
                if hasmask: Y.append(tf.dtypes.cast(tf.image.resize(masks[i], window_shape)[:,:,:1]/255, dtype = 'int32'))
                id = [i]
                idx.append(id)
            elif method == 'rcrop':
                X.append(image)
                if hasmask: Y.append(tf.dtypes.cast(tf.convert_to_tensor(masks[i])[:,:,:1]/255, dtype = 'int32'))
                id = [i]
                idx.append(id)
                
            elif method == 'crop':
                n_y = int(np.ceil(image_size[0]/window_shape[0]))
                n_x = int(np.ceil(image_size[1]/window_shape[1]))
                id = []
                delta_x = int(((n_x*window_shape[1]) - image_size[1])/(n_x-1))
                delta_y = int(((n_y*window_shape[0]) - image_size[0])/(n_y-1))
                for i_y in range(n_y):
                    for i_x in range(n_x):
                        c_i = tf.image.crop_to_bounding_box(image, i_y*(window_shape[0]-delta_y) if i_y < n_y -1 else image_size[0] - window_shape[0],
                                                            i_x*(window_shape[1]-delta_x) if i_x < n_x -1 else image_size[1] - window_shape[1], window_shape[0], window_shape[1])
                        X.append(c_i)
                        if hasmask:
                            c_m = tf.image.crop_to_bounding_box(masks[i], i_y*(window_shape[0]-delta_y) if i_y < n_y -1 else image_size[0] - window_shape[0],
                                                                i_x*(window_shape[1]-delta_x) if i_x < n_x -1 else image_size[1] - window_shape[1], window_shape[0], window_shape[1])
                            Y.append(tf.dtypes.cast(c_m[:,:,:1]/255, dtype = 'int32'))
                        id.append(j)
                        j += 1
                idx.append(id)    
        self.data_dict['X'] = X
        if Y: self.data_dict['Y'] = Y
        self.data_dict['idx'] = idx
        self.data_dict['sizes'] = sizes

    def rejoinMask(self, Y_pred, method = 'crop'):
        sizes = self.data_dict['sizes']
        idx = self.data_dict['idx']
        self.data_dict['Y_pred'] = Y_pred
        masks = []
        window_shape = Y_pred[0].shape
        for i, size in enumerate(sizes):
            if method == 'resize':
                mask = tf.image.resize(Y_pred[i], size, method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                masks.append(mask)
            elif method == 'crop':
                n_y = int(np.ceil(size[0]/window_shape[0]))
                n_x = int(np.ceil(size[1]/window_shape[1]))
                delta_x = int(((n_x*window_shape[1]) - size[1])/(n_x-1))
                delta_y = int(((n_y*window_shape[0]) - size[0])/(n_y-1))
                id = idx[i]
                j = 0
                mask = np.zeros((size[0],size[1],1), dtype=np.float32)
                for i_y in range(n_y):
                    for i_x in range(n_x):
                        if i_y < n_y - 1:
                            offset_y = i_y*(window_shape[0] - delta_y)
                        else:
                            offset_y = size[0] - window_shape[0]
                        if i_x < n_x - 1:
                            offset_x = i_x*(window_shape[1] - delta_x)
                        else:
                            offset_x = size[1] - window_shape[1]
                        mask[offset_y:offset_y+window_shape[0], offset_x:offset_x+window_shape[1]] += np.array(Y_pred[id[j]]).astype(np.float32)
                        j += 1
                for i_y in range(n_y-1):
                    if i_y < n_y - 2:
                        offset_y = (i_y+1)*(window_shape[0] - delta_y)
                        delta_y0 = delta_y
                    else:
                        offset_y = size[0] - window_shape[0]
                        delta_y0 = (i_y+1)*(window_shape[0] - delta_y)  + delta_y - offset_y
                    mask[offset_y:offset_y + delta_y0, :] = mask[offset_y:offset_y+ delta_y0, :]/2
                for i_x in range(n_x - 1):
                    if i_x < n_x - 2:
                        offset_x = (i_x+1)*(window_shape[1] - delta_x)
                        delta_x0 = delta_x
                    else:
                        offset_x = size[1] - window_shape[1]
                        delta_x0 = (i_x+1)*(window_shape[1] - delta_x) + delta_x - offset_x
                    mask[:, offset_x:offset_x+ delta_x0] = mask[:, offset_x:offset_x+ delta_x0]/2
                masks.append(mask)
        self.data_dict['cal_masks'] = masks

    def list2array(self):
        
        list_X =  self.data_dict['X']
        len_X = len(list_X )
        size_X = list_X[0].shape
        X_array = np.zeros((len_X, size_X[0], size_X[1], 3), dtype=np.float32)
        for i in range(len_X):
            X_array[i, :, :, :] = list_X[i]
        if 'Y' in self.data_dict.keys():
            list_Y =  self.data_dict['Y']
            Y_array = np.zeros((len_X, size_X[0], size_X[1], 1), dtype=np.bool_)
            for i in range(len_X):
                Y_array[i, :, :, :] = list_Y[i]
            self.data_dict['Y_array'] = Y_array
        self.data_dict['X_array'] = X_array
        

    def cal_iou(self):
        y_pred = self.data_dict['cal_masks']
        y_true = self.data_dict['masks']
        results = []
        for t in np.arange(0.5, 1, 0.05):
            iou = []
            for y_t, y_p in zip(y_true, y_pred):
                t_y_pred = tf.cast((y_p> t), tf.float32)
                y_t = tf.cast(tf.convert_to_tensor(y_t)[:,:,:1]/255, t_y_pred.dtype)
                intersection = K.sum(K.abs(y_t * t_y_pred), axis=[0,1,2])
                union = K.sum(y_t, axis=[0,1,2]) + K.sum(t_y_pred, axis=[0,1,2]) - intersection
                iou.append((intersection + 1) / (union + 1))
            results.append(K.mean(K.stack(iou), axis=0))
        self.mean_iou = K.mean(K.stack(results), axis=0)

    def display_imgs(self, ix0 = None):
        imageIDs = self.data_dict['imageIDs']
        images = self.data_dict['images']
        masks = self.data_dict['masks']
        preproc = self.data_dict['preprocessed_images']
        if not ix0: ix0 = random.randint(0, len(imageIDs)-1)
        print(f"Image, processed image and mask from data set No. {ix0} with size {images[ix0].size}: {imageIDs[ix0]}")
        display(images[ix0])
        display(tf.keras.preprocessing.image.array_to_img(preproc[ix0]))
        display(masks[ix0])
        
    def display_crops(self, ix0 = None):
        imageIDs = self.data_dict['imageIDs']
        images = self.data_dict['images']
        X = self.data_dict['X']
        Y = self.data_dict['Y']
        idx = self.data_dict['idx']
        preproc = self.data_dict['preprocessed_images']
        if not ix0: ix0 = random.randint(0, len(imageIDs)-1)
        print(f"Processed image and mask from train set No. {ix0} with size {images[ix0].size}: {imageIDs[ix0]}")
        display(tf.keras.preprocessing.image.array_to_img(preproc[ix0]))

        print(f"It crops to {len(idx[ix0])} masks with size {X[ix0].shape}")
        for i in idx[ix0]:

            print(f"cropped image and mask: {i - idx[ix0][0] + 1}")
            display(tf.keras.preprocessing.image.array_to_img(X[i]))
            display(tf.keras.preprocessing.image.array_to_img(Y[i]))
        
    def display_rejoinmasks(self, ix0 = None):
        imageIDs = self.data_dict['imageIDs']
        images = self.data_dict['images']
        images1 = self.data_dict['preprocessed_images']
        cal_masks = self.data_dict['cal_masks']
        masks = self.data_dict['masks']
        if not ix0: ix0 = random.randint(0, len(imageIDs)-1)
        print(f"Image and mask from data set No. {ix0} with size {images[ix0].size}: {imageIDs[ix0]}")
        print(f"Original image")
        display(images[ix0])
        print(f"Processed image")
        display(tf.keras.preprocessing.image.array_to_img(images1[ix0]))
        print(f"Original mask")
        display(masks[ix0])
        print(f"Rejoined mask")
        display(tf.keras.preprocessing.image.array_to_img(cal_masks[ix0]))

        
        