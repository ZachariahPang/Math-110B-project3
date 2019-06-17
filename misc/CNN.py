from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.engine.topology import Network
from keras.layers import *
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np
import os
import random
import scipy.misc
from tqdm import *

DATA_DIR = "data/tiny-imagenet-200"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = 'data/kodak-small'

def load_dataset_small(num_images_per_class_train=200):
    """Loads training and test datasets, from Tiny ImageNet Visual Recogition Challenge.
        
    Arguments:
        num_images_per_class_train: number of images per class to load into training dataset.
        num_images_test: total number of images to load into training dataset.
    """
    X_train = []
    X_test = []
    
    # Create training set.
    for c in os.listdir(TRAIN_DIR):
        c_dir = os.path.join(TRAIN_DIR, c, 'images')
        c_imgs = os.listdir(c_dir)
        random.shuffle(c_imgs)
        for img_name_i in c_imgs[0:num_images_per_class_train]:
            img_i = image.load_img(os.path.join(c_dir, img_name_i))
            x = image.img_to_array(img_i)
            X_train.append(x)
    random.shuffle(X_train)
    
    for c in os.listdir(TEST_DIR):
        img = image.load_img(os.path.join(TEST_DIR, c))
        x = image.img_to_array(img)
        X_test.append(x)
    random.shuffle(X_test)
    
    X_train = np.array(X_train)/255.
    X_test = np.array(X_test)/255.
    # Return train and test data as numpy arrays.
    return X_train, X_test

# Variable used to weight the losses of the secret and cover images (See paper for more details)
beta = 1.0
    
# Loss for reveal network
def rev_loss(s_true, s_pred):
    # Loss for reveal network is: beta * |S-S'|
    if s_true.shape[-1] == 6:
        s_true_1 = s_true[...,0:3]
        s_true_2 = s_true[...,3:6]
        s_pred_1 = s_pred[...,0:3]
        s_pred_2 = s_pred[...,3:6]
        return 2 * beta * K.sum(K.square(s_true_1 - s_pred_1) + K.square(s_true_2 - s_pred_2))
    else:
        return beta * K.sum(K.square(s_true - s_pred))

# Loss for the full model, used for preparation and hidding networks
def full_loss(y_true, y_pred):
    # Loss for the full model is: |C-C'| + beta * |S-S'|
    if y_true.shape[-1] == 9:
        s_true, c_true = y_true[...,0:6], y_true[...,6:9]
        s_pred, c_pred = y_pred[...,0:6], y_pred[...,6:9]
    else:
        s_true, c_true = y_true[...,0:3], y_true[...,3:6]
        s_pred, c_pred = y_pred[...,0:3], y_pred[...,3:6]
    
    s_loss = rev_loss(s_true, s_pred)
    c_loss = K.sum(K.square(c_true - c_pred))
    
    return s_loss + c_loss


# Returns the encoder as a Keras model, composed by Preparation and Hiding Networks.
def make_encoder(multi,prep):
    if multi:
        input_size_C = (64,64,3)
        input_size_S = (64,64,6)
    else :
        input_size_C = (64,64,3)
        input_size_S = (64,64,3)

    input_S = Input(shape=(input_size_S))
    input_C= Input(shape=(input_size_C))

    # Preparation Network
    x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_prep0_3x3')(input_S)
    x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_prep0_4x4')(input_S)
    x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_prep0_5x5')(input_S)
    x = concatenate([x3, x4, x5])
    
    x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_prep1_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_prep1_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_prep1_5x5')(x)
    x = concatenate([x3, x4, x5])
    
    x = concatenate([input_C, x])
    
    # Hiding network
    x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_hid0_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_hid0_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_hid0_5x5')(x)
    x = concatenate([x3, x4, x5])
    
    x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_hid1_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_hid1_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_hid1_5x5')(x)
    x = concatenate([x3, x4, x5])
    
    x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_hid2_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_hid2_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_hid2_5x5')(x)
    x = concatenate([x3, x4, x5])
    
    x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_hid3_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_hid3_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_hid3_5x5')(x)
    x = concatenate([x3, x4, x5])
    
    x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_hid4_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_hid4_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_hid5_5x5')(x)
    x = concatenate([x3, x4, x5])
    
    output_Cprime = Conv2D(3, (3, 3), strides = (1, 1), padding='same', activation='relu', name='output_C')(x)
    
    return Model(inputs=[input_S, input_C],
                 outputs=output_Cprime,
                 name = 'Encoder')

# Returns the decoder as a Keras model, composed by the Reveal Network
def make_decoder(multi, fixed=False):
    input_size = (64,64,3)

    # Reveal network
    reveal_input = Input(shape=(input_size))
    
    # Adding Gaussian noise with 0.01 standard deviation.
    input_with_noise = GaussianNoise(0.01, name='output_C_noise')(reveal_input)
    
    x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev0_3x3')(input_with_noise)
    x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev0_4x4')(input_with_noise)
    x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev0_5x5')(input_with_noise)
    x = concatenate([x3, x4, x5])
    
    x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev1_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev1_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev1_5x5')(x)
    x = concatenate([x3, x4, x5])
    
    x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev2_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev2_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev2_5x5')(x)
    x = concatenate([x3, x4, x5])
    
    x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev3_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev3_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev3_5x5')(x)
    x = concatenate([x3, x4, x5])
    
    x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev4_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev4_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev5_5x5')(x)
    x = concatenate([x3, x4, x5])
    
    if multi:
        output_Sprime = Conv2D(6, (3, 3), strides = (1, 1), padding='same', activation='relu', name='output_S')(x)
    else:
        output_Sprime = Conv2D(3, (3, 3), strides = (1, 1), padding='same', activation='relu', name='output_S')(x)
    
    if not fixed:
        return Model(inputs=reveal_input,
                     outputs=output_Sprime,
                     name = 'Decoder')
    else:
        return Network(inputs=reveal_input,
                         outputs=output_Sprime,
                         name = 'DecoderFixed')

def lr_schedule(epoch_idx):
    if epoch_idx < 200:
        return 0.001
    elif epoch_idx < 400:
        return 0.0003
    elif epoch_idx < 600:
        return 0.0001
    else:
        return 0.00003

# Full model.
class model(object):
    def __init__(self, multi, prep):
        self.multi = multi
        self.prep = prep

    def make_model(self):
        if self.multi :
            input_size_C = (64,64,3)
            input_size_S = (64,64,6)
        else :
            input_size_C = (64,64,3)
            input_size_S = (64,64,3)

        input_S = Input(shape=(input_size_S))
        input_C= Input(shape=(input_size_C))
        
        encoder = make_encoder(self.multi,self.prep)
        
        decoder = make_decoder(self.multi, fixed=False)
        decoder.compile(optimizer='adam', loss=rev_loss)
        decoder.trainable = False
        
        output_Cprime = encoder([input_S, input_C])
        output_Sprime = decoder(output_Cprime)

        autoencoder = Model(inputs=[input_S, input_C],
                            outputs=concatenate([output_Sprime, output_Cprime]))
        autoencoder.compile(optimizer='adam', loss=full_loss)
        
        self.encoder = encoder 
        self.decoder = decoder 
        self.autoencoder = autoencoder



    def train(self, input_S, input_C, NB_EPOCHS=1000, BATCH_SIZE =32):

        m = input_S.shape[0]
        loss_history = []
        log = tqdm_notebook(range(NB_EPOCHS),ncols=800)
        for epoch in log:
            np.random.shuffle(input_S)
            np.random.shuffle(input_C)

            t = range(0, input_S.shape[0], BATCH_SIZE)
            ae_loss = []
            rev_loss = []
            for idx in t:

                batch_S = input_S[idx:min(idx + BATCH_SIZE, m)]
                batch_C = input_C[idx:min(idx + BATCH_SIZE, m)]

                C_prime = self.encoder.predict([batch_S, batch_C])

                ae_loss.append(self.autoencoder.train_on_batch(x=[batch_S, batch_C],
                                                        y=np.concatenate((batch_S, batch_C),axis=3)))
                rev_loss.append(self.reveal.train_on_batch(x=C_prime,
                                                    y=batch_S))

                # Update learning rate
                K.set_value(self.autoencoder.optimizer.lr, lr_schedule(epoch))
                K.set_value(self.reveal.optimizer.lr, lr_schedule(epoch))
            log.set_description('Epoch {} | Loss AE {:10.2f} | Loss Rev {:10.2f}'.format(epoch + 1, np.mean(ae_loss), np.mean(rev_loss)))
            loss_history.append(np.mean(ae_loss))
            # Plot loss through epochs
        plt.plot(loss_history)
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

