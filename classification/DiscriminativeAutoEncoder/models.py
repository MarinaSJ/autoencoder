from tensorflow.keras.optimizers import Adam
import prepdata.discriminant_methods as dm
from tensorflow.keras.layers import Input, Dense, GaussianDropout, Dropout
from tensorflow.keras.models import Model
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import os

from sklearn.utils import class_weight

#from saphyra import *

import pickle

import numpy as np
np.random.seed(43)


# Discrimination methods
def move_to_means(train_data, train_classes, a=0.2):
    '''

    Discrimination method that moves the data points towards the mean of their class.

    '''
    n_classes = np.max(train_classes)+1
    n_classes = int(n_classes)
    print('------------------- n_classes : ', n_classes,
          ' a : ', a, ' -----------------------')
    centers = np.zeros((n_classes, train_data.shape[1]))
    new_train_data = np.zeros(train_data.shape)
    for i in range(n_classes):
        indices = np.squeeze(np.where(train_classes == i))
        centers[i, ...] = np.mean(train_data[indices, ...], axis=0)
    for i in range(train_data.shape[0]):
        new_train_data[i, ...] = (
            1-a)*train_data[i, ...] + a*centers[train_classes[i], ...]
    return new_train_data

def move_away_from_means(train_data, train_classes, a=0.1, k=3):
    n_classes = np.max(train_classes)+1
    n_classes = int(n_classes)
    centers = np.zeros((n_classes, train_data.shape[1]))
    for j in range(n_classes):
        indices = np.squeeze(np.where(train_classes == j))
        centers[j, ...] = np.mean(train_data[indices, ...], axis=0)
    new_train_data = np.zeros(train_data.shape)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', n_jobs=-1)
    nbrs.fit(centers)
    n_ind = nbrs.kneighbors(train_data, n_neighbors=k + 1, return_distance=False)
    for i in range(train_data.shape[0]):
        new_train_data[i,...] = (1-a)*train_data[i,...] + a*np.squeeze(np.mean(centers[n_ind[i]][1:]))
    return new_train_data


# Reconstruction networks
def create_autoencoder(input_dim=784, layers=[100], denoising=True, p=0.1):
    '''

    Returns the autoencoder and its encoder model.

    '''

    print('Input dimension - %d' % input_dim)
    input = Input(shape=(input_dim,))
    if (denoising):
        # x = GaussianNoise(p)(input)
        x = Dropout(p)(input)
    else:
        x = input
    for idx, l in enumerate(layers[:-1]):
        print('Creating layer with %d' % l)
        x = Dense(l, activation='relu',
                  kernel_initializer='glorot_normal', name='fc'+str(idx))(x)
    print('Creating layer previous with %d' % layers[-1])
    encoded = Dense(layers[-1], activation='relu',
                    kernel_initializer='glorot_normal', name='fc'+str(len(layers)))(x)
    x = encoded
    for idx, l in enumerate(layers[::-1][1:]):
        print('Creating layer with %d' % l)
        x = Dense(l, activation='relu', kernel_initializer='glorot_normal')(x)
    print('Input dimension - %d' % input_dim)
    output = Dense(input_dim, activation='linear',
                   kernel_initializer='glorot_normal')(x)
    autoencoder = Model(inputs=input, outputs=output)
    encoder = Model(inputs=input, outputs=encoded)

    return autoencoder, encoder

# Classification networks


def create_classifier(input_dim=784, n_classes=10, layers=[100], weights_file=None, denoising=True, p=0.1):
    '''

    A simple MLP classifier with 1 Gaussian Dropout layer.

    '''

    print('Input dimension - %d' % input_dim)
    input = Input(shape=(input_dim,))
    if (denoising):
        # x = GaussianNoise(p)(input)
        x = GaussianDropout(p)(input)
    else:
        x = input
    for idx, l in enumerate(layers):
        print('Creating layer with %d' % l)
        x = Dense(l, activation='sigmoid',
                  kernel_initializer='glorot_normal', name='fc'+str(idx))(x)
    # x = Dense(128)(x)
    output = Dense(n_classes, activation='softmax',
                   kernel_initializer='glorot_normal')(x)
    classifier = Model(inputs=input, outputs=output)
    if weights_file is not None:
        classifier.load_weights(weights_file, by_name=True)
    return classifier


class DiscriminativeAutoEncoder():
    def __init__(
        self,
        input_shape: int = 784,  # for mnist and greece 784
        layers: list = [100],  # 100 standart of greece
        denoising: bool = True,
        p: float = 0.1,  # Gausian dropout
        classes_num: int = 10,  # For Greece 10, for electrons jets 2, for boosted 3
        file_path: str = './spits'
    ):

        self.input_dim = input_shape
        self.n_classes = classes_num

        self.path = file_path
        self.pklpath = os.path.join(self.path, 'pickle/')
        self.mdlpath = os.path.join(self.path, 'saved_models/')

        # defines the classifier model used
        self.model_clf = create_classifier(
            input_dim=layers[-1], layers=[], denoising=False, p=p, n_classes=classes_num)
        self.model_rct, self.model_ecd = create_autoencoder(
            input_dim=input_shape, layers=layers, denoising=denoising, p=p)  # defines which autoencoder model used
        self.learning_rate = 0.002
        self.loss = 'mse'
        
        self.is_trained = False
        
        folders = self.path.split('/')
        for i in range(len(folders)):
            if i == 0:
                continue
            folder_path = '.'+os.path.join('.',folders[0],'/', *folders[1:i+1])
            try:
                os.mkdir(folder_path)
            except OSError as error:
                print(error)
        
        try: 
            os.mkdir(self.path)
        except OSError as error: 
            print(error)
        try:
            os.mkdir(self.pklpath)
        except OSError as error: 
            print(error)
        try:
            os.mkdir(self.mdlpath)
        except OSError as error: 
            print(error)

    def fit(self, train_x, train_y, 
            validation_data, #(x_val, y_val)
            epoch = None,
            batch_size= 128,
            verbose = 2,
            callbacks = None,
            class_weight = None,
            shuffle = True,
            filename = 'model'
            ):
        '''
        As classes devem ser numéricas (ex: 1,2,3,4...) e não maximamente esparsas!
        '''
        # super().fit()
        print('DiscriminateAutoEncoder.fit Function')



        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                          patience=3)
        early_stopping_1 = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                            min_delta=0,
                                                            patience=5,
                                                            verbose=0)


        test_x, test_y = validation_data
        
        train_accuracy = np.zeros((5))
        test_accuracy = np.zeros((5))
        loss = []
        means = 'to_method'  # or 'from_method'
        self.compile()  # (optimizer=Adam(lr=0.0002), loss='mse')
        # self.model_ecd.compile(optimizer=Adam(lr=0.0002), loss='mse')

        class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_y),
                                                 train_y)

        for epoch in range(5):
            if epoch == 0:
                new_train_x = train_x
            print('=========== Reconstruction Model Fit ===========')
            # 600 na primeira epoca [ 600, 300, 300, 150, 150 ]  |  [ 4, 4, 4, 2, 2 ]
            hist = self.model_rct.fit(train_x, new_train_x, epochs=[600, 300, 300, 150, 150][epoch],
                                      batch_size=128, verbose=2, callbacks=[early_stopping], class_weight=class_weights)
            self.model_rct.save_weights(
                '{path}{model}autoencoder_weights_epoch_{epoch}.h5'.format(epoch=epoch,model=filename,path=self.mdlpath))
            self.model_rct.save(
                '{path}{model}autoencoder_weights_epoch_{epoch}.keras'.format(epoch=epoch,model=filename,path=self.mdlpath))
            print('Autoencoder model saved on {path}{model}autoencoder_weights_epoch_{epoch}'.format(epoch=epoch,model=filename,path=self.mdlpath))

            # self.model_rct.summary()

            train_recon = self.model_rct.predict(train_x)
            test_recon = self.model_rct.predict(test_x)
            train_hidden = self.model_ecd.predict(train_x)
            if epoch == 0:
                new_train_hidden = train_hidden
            test_hidden = self.model_ecd.predict(test_x)

            print('=========== Classification Model Fit ===========')
            cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False,
                                                                    label_smoothing=0.0,
                                                                    reduction="auto",
                                                                    name="categorical_crossentropy",
                                                                    )
            
            self.model_clf.compile(optimizer=Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                                   loss=cross_entropy, metrics=['accuracy', "categorical_crossentropy"])
            
            if self.n_classes == 1:
                
                classifer_hist = self.model_clf.fit(train_hidden, train_y, batch_size=64,  # 128 -> problem 64
                                                    epochs=150,  # 150
                                                    verbose=2,
                                                    validation_data=(test_hidden, test_y),#[test_hidden, to_categorical(test_y)],
                                                    callbacks=[early_stopping, early_stopping_1], class_weight=class_weights)
            else:
                classifer_hist = self.model_clf.fit(train_hidden, to_categorical(train_y), batch_size=64,  # 128 -> problem 64
                                                    epochs=150,  # 150
                                                    verbose=2,
                                                    validation_data=(test_hidden, to_categorical(test_y)),#[test_hidden, to_categorical(test_y)],
                                                    callbacks=[early_stopping, early_stopping_1], class_weight=class_weights)
            self.model_clf.save_weights(
                '{path}{model}clf_autoencoder_weights_epoch_{epoch}.h5'.format(epoch=epoch,model=filename,path=self.mdlpath))
            self.model_clf.save(
                '{path}{model}clf_autoencoder_weights_epoch_{epoch}.keras'.format(epoch=epoch,model=filename,path=self.mdlpath))
            #print(f'Classification model saved on {path}clf_autoencoder_weights_epoch_{epoch}'.format(epoch=epoch,model=filename,path=self.path))

            train_recon = self.model_rct.predict(train_x)
            test_recon = self.model_rct.predict(test_x)

            # share autoencode weights with encoder model
            train_hidden = self.model_ecd.predict(train_x)

            if epoch == 0:
                new_train_hidden = train_hidden
            test_hidden = self.model_ecd.predict(test_x)

            if means == 'to_method':
                new_train_x = move_to_means(train_x, train_y, 0.4) #active
                #19/09/2023 : será que não deveriamos colocar train_hidden na linha acima?
            if means == 'from_method':
                new_train_x = move_away_from_means(new_train_x, train_y, 0.1)

            print('=========== Saving History ===========')
            rec_hist = hist.history
            clf_hist = classifer_hist.history
            with open('{path}{model}hists_{epoch}.pickle'.format(epoch=epoch,model=filename, path=self.pklpath), 'wb') as f:
                # save the history of the classifier and autoencoder fit
                pickle.dump([rec_hist, clf_hist], f)
            print(
                'Historys saved on ./spits/pickle/{model}hists_{epoch}.pickle'.format(epoch=epoch,model=filename))
        print('=========== Done ===========')
        self.is_trained = True
        return classifer_hist

    def predict(self, data):

        print('DiscriminateAutoEncoder.predict Function')

        rct_predict = self.model_rct.predict(data)
        ecd_predict = self.model_ecd.predict(data)
        clf_predict = self.model_clf.predict(ecd_predict)

        return self.model_clf.predict(self.model_ecd.predict(data))

    def compile(self,
                loss = None,
                metrics=None):

        print('DiscriminateAutoEncoder.compile Function')
        self.model_rct.compile(optimizer=Adam(lr=self.learning_rate), loss=self.loss, metrics=[tf.keras.metrics.MeanSquaredError(name="mean_squared_percentage_error",
                                                                                                                                 dtype=None)
                                                                                               ])
        self.model_ecd.compile(optimizer=Adam(
            lr=self.learning_rate), loss=self.loss)

        return self.model_clf.compile(optimizer=Adam(lr=self.learning_rate), loss=self.loss)

    def summary(self):

        print('DiscriminateAutoEncoder.summary Function')

        return {self.model_rct.summary(),
                self.model_clf.summary()}

    def to_json(self):

        print('DiscriminateAutoEncoder.to_json Function')

        self.model_rct.to_json()
        self.model_ecd.to_json()

        return self.model_clf.to_json()

    def get_weights(self):
        # marcar os argumentos!

        print('DiscriminateAutoEncoder.get_weights Function')

        self.model_rct.get_weights()
        self.model_ecd.get_weights()

        return self.model_clf.get_weights()

    def save_weights(self, filepath=None, model_name: str = "model"):

        print('DiscriminateAutoEncoder.save_weights Function')
        
        if self.is_trained:
            if filepath == None:
                filepath = self.mdlpath

            self.model_rct.save_weights(filepath + model_name + '_rec.h5')
            self.model_ecd.save_weights(filepath + model_name + '_ecd.h5')
            self.model_clf.save_weights(filepath + model_name + '_clf.h5')
            return 0
        else:
            print('no model is trained')
        return -1

    def save(self, filepath=None, model_name: str = "model"):
            
        print('DiscriminateAutoEncoder.save Function')
        if self.is_trained:
            if filepath == None:
                filepath = self.mdlpath

            self.model_rct.save(filepath + model_name + '_rec.keras')
            self.model_ecd.save(filepath + model_name + '_ecd.keras')
            self.model_clf.save(filepath + model_name + '_clf.keras')
            return 0
        
        else:
            print('no model is trained')
        return -1
    def load(self, filepath=None, model_name: str = "model"):
            
        print('DiscriminateAutoEncoder.load Function')
        if not self.is_trained:
            if filepath == None:
                filepath = self.mdlpath
                
            self.model_rct = tf.keras.models.load_model(filepath + model_name + '_rec.keras')
            self.model_ecd = tf.keras.models.load_model(filepath + model_name + '_ecd.keras')
            self.model_clf = tf.keras.models.load_model(filepath + model_name + '_clf.keras')
            self.is_trained = True
            return 0
        
        else:
            print('The model is trained')
