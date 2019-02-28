# usage: python driving_models.py 1 - train the dave-orig model

from __future__ import print_function

import sys
from keras.models import Sequential
from keras.layers import Convolution2D, Input, Dense, Flatten, Lambda, MaxPooling2D, Dropout , Activation, SpatialDropout2D, merge
from keras import models, optimizers, backend
from keras.layers import core, convolutional, pooling, TimeDistributed, LSTM
from sklearn import model_selection
from configs import bcolors
from data_utils import load_train_data, load_test_data
from keras.optimizers import SGD
from keras.regularizers import l2
from utils import *
from collections import deque
from keras.models import model_from_json


def Dave_v2(input_tensor=None, load_weights=True):
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, init = normal_init, subsample= (2, 2), name='conv1_1', input_shape=(66, 200, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, init = normal_init, subsample= (2, 2), name='conv2_1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, init = normal_init, subsample= (2, 2), name='conv3_1'))
    model.add(Activation('relu'))  
    model.add(Convolution2D(64, 3, 3, init = normal_init, subsample= (1, 1), name='conv4_1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, init = normal_init, subsample= (1, 1), name='conv4_2'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1164, init = normal_init, name = "dense_0"))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Dense(100, init = normal_init,  name = "dense_1"))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Dense(50, init = normal_init, name = "dense_2"))
    model.add(Activation('relu'))
    #model.add(Dropout(p))
    model.add(Dense(10, init = normal_init, name = "dense_3"))
    model.add(Activation('relu'))
    model.add(Dense(1, init = normal_init, name = "dense_4"))
    model.add(Lambda(atan_layer, output_shape = atan_layer_shape, name = "prediction"))         #######
    model.load_weights('./models/dave2/dave2.hdf5') # it means the weights have been loaded
    model.compile(loss = 'mse', optimizer = 'Adam')
    return model

def Dave_orig(input_tensor=None, load_weights=False):  # original dave
    if input_tensor is None:
        input_tensor = Input(shape=(100, 100, 3))
    x = Convolution2D(24, (5, 5), padding='valid', activation='relu', strides=(2, 2), name='block1_conv1')(input_tensor)
    x = Convolution2D(36, (5, 5), padding='valid', activation='relu', strides=(2, 2), name='block1_conv2')(x)
    x = Convolution2D(48, (5, 5), padding='valid', activation='relu', strides=(2, 2), name='block1_conv3')(x)
    x = Convolution2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1), name='block1_conv4')(x)
    x = Convolution2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1), name='block1_conv5')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(1164, activation='relu', name='fc1')(x)
    x = Dense(100, activation='relu', name='fc2')(x)
    x = Dense(50, activation='relu', name='fc3')(x)
    x = Dense(10, activation='relu', name='fc4')(x)
    x = Dense(1, name='before_prediction')(x)
    x = Lambda(atan_layer, output_shape=atan_layer_shape, name='prediction')(x)

    m = Model(input_tensor, x)
    if load_weights:
        m.load_weights('./Model1.h5')

    # compiling
    m.compile(loss='mse', optimizer='adadelta')
    print(bcolors.OKGREEN + 'Model compiled' + bcolors.ENDC)
    return m
def One_to_radius(x):
    return tf.multiply(x,math.pi)
def Dave_v3(input_tensor = None,load_weights = False):
    model = models.Sequential()
    model.add(convolutional.Convolution2D(16, 3, 3, input_shape=(32, 128, 3), activation='relu'))
    model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
    model.add(convolutional.Convolution2D(32, 3, 3, activation='relu'))
    model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
    model.add(convolutional.Convolution2D(64, 3, 3, activation='relu'))
    model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
    model.add(core.Flatten())
    model.add(core.Dense(500, activation='relu'))
    #model.add(core.Dropout(.5))
    model.add(core.Dense(100, activation='relu'))
    #model.add(core.Dropout(.25))
    model.add(core.Dense(20, activation='relu'))
    model.add(core.Dense(1))
    model.add(Lambda(One_to_radius, output_shape = atan_layer_shape, name = "prediction"))
    if load_weights:
        model.load_weights('./models/dave3/dave3.h5')
    model.compile(optimizer=optimizers.Adam(lr=1e-04), loss='mean_squared_error')
    return model

def Dave_norminit(input_tensor=None, load_weights=False):  # original dave with normal initialization
    if input_tensor is None:
        input_tensor = Input(shape=(100, 100, 3))
    x = Convolution2D(24, (5, 5), padding='valid', activation='relu', strides=(2, 2),
                      name='block1_conv1')(input_tensor)
    x = Convolution2D(36, (5, 5), padding='valid', activation='relu', strides=(2, 2),
                      name='block1_conv2')(x)
    x = Convolution2D(48, (5, 5), padding='valid', activation='relu', strides=(2, 2),
                      name='block1_conv3')(x)
    x = Convolution2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1),
                      name='block1_conv4')(x)
    x = Convolution2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1),
                      name='block1_conv5')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(1164, kernel_initializer=normal_init, activation='relu', name='fc1')(x)
    x = Dense(100, kernel_initializer=normal_init, activation='relu', name='fc2')(x)
    x = Dense(50, kernel_initializer=normal_init, activation='relu', name='fc3')(x)
    x = Dense(10, kernel_initializer=normal_init, activation='relu', name='fc4')(x)
    x = Dense(1, name='before_prediction')(x)
    x = Lambda(atan_layer, output_shape=atan_layer_shape, name='prediction')(x)

    m = Model(input_tensor, x)
    if load_weights:
        m.load_weights('./Model2.h5')

    # compiling
    m.compile(loss='mse', optimizer='adadelta')
    print(bcolors.OKGREEN + 'Model compiled' + bcolors.ENDC)
    return m

def Dave_dropout(input_tensor=None, load_weights=False):  # simplified dave
    if input_tensor is None:
        input_tensor = Input(shape=(100, 100, 3))
    x = Convolution2D(16, (3, 3), padding='valid', activation='relu', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)
    x = Convolution2D(32, (3, 3), padding='valid', activation='relu', name='block1_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool2')(x)
    x = Convolution2D(64, (3, 3), padding='valid', activation='relu', name='block1_conv3')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool3')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(500, activation='relu', name='fc1')(x)
    x = Dropout(.5)(x)
    x = Dense(100, activation='relu', name='fc2')(x)
    x = Dropout(.25)(x)
    x = Dense(20, activation='relu', name='fc3')(x)
    x = Dense(1, name='before_prediction')(x)
    x = Lambda(atan_layer, output_shape=atan_layer_shape, name="prediction")(x)

    m = Model(input_tensor, x)
    if load_weights:
        m.load_weights('./Model3.h5')

    # compiling
    m.compile(loss='mse', optimizer='adadelta')
    print(bcolors.OKGREEN + 'Model compiled' + bcolors.ENDC)
    return m

def rmse(y_true, y_pred):
    '''Calculates RMSE
    '''
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# def chauffeur(input_tensor=None, load_weights=True):
#     '''
#     Creates an LstmModel using a model in the input_model_config

#     @param model_path - s3 uri to save the model
#     @param input_shape - timestepped shape (timesteps, feature dims)
#     @param batch_input_shape - (batch_size, feature dims)
#     @param timesteps - timesteps inclusive of the current frame
#                         (10 - current frame + 9 previous frames)
#     @param loss - loss function on the model
#     @param W_l2 - W_l2 regularization param
#     @param metrics - metrics to track - (rmse, mse...)
#     @param scale - factor by which to scale the labels
#     '''
    
#     input_shape = (10, 120, 320, 3)
#     W_l2 = 0.001
#     loss = 'mean_squared_error'

#     model = Sequential()
#     model.add(TimeDistributed(Convolution2D(24, 5, 5,
#         init= "he_normal",
#         activation='relu',
#         subsample=(5, 4),
#         border_mode='valid'), input_shape=input_shape))
#     model.add(TimeDistributed(Convolution2D(32, 5, 5,
#         init= "he_normal",
#         activation='relu',
#         subsample=(3, 2),
#         border_mode='valid')))
#     model.add(TimeDistributed(Convolution2D(48, 3, 3,
#         init= "he_normal",
#         activation='relu',
#         subsample=(1,2),
#         border_mode='valid')))
#     model.add(TimeDistributed(Convolution2D(64, 3, 3,
#         init= "he_normal",
#         activation='relu',
#         border_mode='valid')))
#     model.add(TimeDistributed(Convolution2D(128, 3, 3,
#         init= "he_normal",
#         activation='relu',
#         subsample=(1,2),
#         border_mode='valid')))
#     model.add(TimeDistributed(Flatten()))
#     model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
#     model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
#     model.add(LSTM(64, dropout_W=0.2, dropout_U=0.2))
#     model.add(Dropout(0.2))
#     model.add(Dense(
#         units=256,
#         init='he_normal',
#         activation='relu',
#         kernel_regularizer=l2(W_l2)))
#     model.add(Dropout(0.2))
#     model.add(Dense(
#         units=1,
#         init='he_normal',
#         kernel_regularizer=l2(W_l2), name='before_prediction'))
#     model.add(Lambda(atan_layer, output_shape = atan_layer_shape, name = "prediction"))
#     model.compile(loss=loss, optimizer='adadelta', metrics=[rmse])
#     # print("+++++++++++++++++++++++++++++++++++++++++++++")
#     # if load_weights:
#     #     model.load_weights('./models/chauffeur/lstm.weights')
#         # print("LOAD FINISH")
#     # print("=============================================")
#     return model

class ChauffeurModel(object):
    def __init__(self,
                cnn_json_path='./models/chauffeur/cnn.json',
                cnn_weights_path='./models/chauffeur/cnn.weights',
                lstm_json_path='./models/chauffeur/lstm.json',
                lstm_weights_path='./models/chauffeur/lstm.weights'):

        self.cnn = self.load_from_json(cnn_json_path, cnn_weights_path)
        self.encoder = self.load_encoder(cnn_json_path, cnn_weights_path)
        self.lstm = self.load_from_json(lstm_json_path, lstm_weights_path)

        self.scale = 16.
        self.timesteps = 100


        self.timestepped_x = np.empty((1, self.timesteps, 8960))

    def load_encoder(self, cnn_json_path, cnn_weights_path):
        model = self.load_from_json(cnn_json_path, cnn_weights_path)
        model.load_weights(cnn_weights_path)

        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []

        return model

    def load_from_json(self, json_path, weights_path):
        model = model_from_json(open(json_path, 'r').read())
        model.load_weights(weights_path)
        return model

    def make_cnn_only_predictor(self):
        def predict_fn(img):
            img = cv2.resize(img, (320, 240))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img = img[120:240, :, :]
            img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
            img = ((img-(255.0/2))/255.0)

            return self.cnn.predict_on_batch(img.reshape((1, 120, 320, 3)))[0, 0] / self.scale

        return predict_fn

    def make_stateful_predictor(self):
        steps = deque()

        def predict_fn(img):
            # preprocess image to be YUV 320x120 and equalize Y histogram
            img = cv2.resize(img, (320, 240))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img = img[120:240, :, :]
            img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
            img = ((img-(255.0/2))/255.0)

            # apply feature extractor
            img = self.encoder.predict_on_batch(img.reshape((1, 120, 320, 3)))

            # initial fill of timesteps
            if not len(steps):
                for _ in range(self.timesteps):
                    steps.append(img)

            # put most recent features at end
            steps.popleft()
            steps.append(img)

            timestepped_x = np.empty((1, self.timesteps, img.shape[1]))
            for i, img in enumerate(steps):
                timestepped_x[0, i] = img

            return self.lstm.predict_on_batch(timestepped_x)[0, 0] / self.scale

        return predict_fn

def calc_rmse(yhat, label):
    mse = 0.
    count = 0
    if len(yhat) != len(label):
        print ("yhat and label have different lengths")
        return -1
    for i in range(len(yhat)):
        count += 1
        predicted_steering = yhat[i]
        steering = label[i]
        #print(predicted_steering)
        #print(steering)
        mse += (float(steering) - float(predicted_steering))**2.
    return (mse/count) ** 0.5



if __name__ == '__main__':
    # train the model
    batch_size = 256
    nb_epoch = 10
    model_name = sys.argv[1]

    if model_name == '1':
        model = Dave_orig()
        save_model_name = './Model1.h5'
    elif model_name == '2':
        # K.set_learning_phase(1)
        model = Dave_norminit()
        save_model_name = './Model2.h5'
    elif model_name == '3':
        # K.set_learning_phase(1)
        model = Dave_dropout()
        save_model_name = './Model3.h5'
    else:
        print(bcolors.FAIL + 'invalid model name, must one of 1, 2 or 3' + bcolors.ENDC)

    # the data, shuffled and split between train and test sets
    train_generator, samples_per_epoch = load_train_data(batch_size=batch_size, shape=(100, 100))

    # trainig
    model.fit_generator(train_generator,
                        steps_per_epoch=math.ceil(samples_per_epoch * 1. / batch_size),
                        epochs=nb_epoch,
                        workers=8,
                        use_multiprocessing=True)
    print(bcolors.OKGREEN + 'Model trained' + bcolors.ENDC)

    # evaluation
    K.set_learning_phase(0)
    test_generator, samples_per_epoch = load_test_data(batch_size=batch_size, shape=(100, 100))
    model.evaluate_generator(test_generator,
                             steps=math.ceil(samples_per_epoch * 1. / batch_size))
    # save model
    model.save_weights(save_model_name)
