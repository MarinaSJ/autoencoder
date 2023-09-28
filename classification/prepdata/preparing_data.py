'''

This model prepares the data that will be used to train the autoencoder.

'''

from tensorflow.keras.datasets import mnist # test dataset to more quickly test the code

from prepdata.datafilter import *
import tensorflow as tf

import numpy as np
import gzip 
import pickle as cPickle
from sklearn.model_selection import StratifiedKFold



def norm1( data ):
    '''
    
    Normalizes the data by the sum of the absolute values of the data.
    
    '''
    
    norms = np.abs( data.sum(axis=1) )
    norms[norms==0] = 1
    
    return data/norms[:,None]


def check_extension( filename , extension):
    '''
    
    Checks the extension of the file is the same as the one given.
    
    '''
    
    extensions = extension.split('|')
    for ext in extensions:
        if filename.endswith("."+ext):
            return True
    return False


def loadfile( filename:str, allow_pickle = True ):
    '''
    
    Loads the file from the following path 
    converting extensions: npz, pic.gz, pic. to csv
    
    '''

    if check_extension(filename, 'npz'):
        return dict(np.load(filename, allow_pickle=allow_pickle))
    elif check_extension(filename, 'pic.gz'):
        f = gzip.GzipFile(filename, 'rb')
        o = cPickle.load(f)
        f.close()
        return o
    elif check_extension(filename, 'pic'):
        return None
    else:
        return None


def splits(data, target):
    
    '''
    
    Divide the data into 10 folds.
    Returns a dataframe with the data and the targets and 10 columns with the folds.
    On each column, if it used for training == True, else == False.

    '''
    df = pd.DataFrame(data)
    df['targets'] = target

    cv = StratifiedKFold(n_splits=10, random_state=512, shuffle=True)

    splits = [(train_index, val_index) for train_index, val_index in cv.split(data,target)]

    folds_list = np.zeros(data.shape[0])

    for fold in range(10):
        for idx in splits[fold][0]:
            folds_list[idx] = True
            
        for idx in splits[fold][1]:
            folds_list[idx] = False
        
        folds_list = folds_list.astype(bool)

        df[f'is_train_fold_{fold}'] = folds_list

    return df


def train_test(df, fold):
    
    '''
    
    splits the data into training and validation sets for a given fold.

    '''

    print('---------------spliting concatenated data into train and val...---------------')
    x_train = df[df[f'is_train_fold_{fold}'] == True].drop(['targets','is_train_fold_0','is_train_fold_1','is_train_fold_2','is_train_fold_3','is_train_fold_4','is_train_fold_5','is_train_fold_6','is_train_fold_7','is_train_fold_8','is_train_fold_9'], axis=1).to_numpy()
    y_train = df[df[f'is_train_fold_{fold}'] == True]['targets'].to_numpy().astype(int)

    x_val = df[df[f'is_train_fold_{fold}'] == False].drop(['targets','is_train_fold_0','is_train_fold_1','is_train_fold_2','is_train_fold_3','is_train_fold_4','is_train_fold_5','is_train_fold_6','is_train_fold_7','is_train_fold_8','is_train_fold_9'], axis=1).to_numpy()
    y_val = df[df[f'is_train_fold_{fold}'] == False]['targets'].to_numpy().astype(int)

    return x_train, y_train, x_val, y_val


def training_splits(path, cv, procedure='rawdata', sort=0):
    '''
    
    Splits the data into training and validation sets.
    
    '''
    
    if path == None or procedure == 'mnist':
        d = mnist.load_data()
        (x_train, y_train), (x_val, y_val) = mnist.load_data()
        x_train = x_train.astype('float32') / 255.
        x_val = x_val.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_val = x_val.reshape((len(x_val), np.prod(x_val.shape[1:])))
        
    else:
        d = loadfile(path)
        
        if procedure == 'rawdata' or procedure == None:
            reduction_of_rings = 1
            
            if reduction_of_rings == 1: #normalization considering all rings
                data = norm1(d['data'][:,1:101])
                
            if reduction_of_rings > 1: #normalization considering only in half of rings
            
                data_df, selection_list = rings_reduction(reduction_of_rings,d)
                data = norm1(data_df[selection_list].values)
            
            target = d['target']
            target[target!=1]=-1
            splits = [(train_index, val_index) for train_index, val_index in cv.split(data,target)]

            x_train = data [ splits[sort][0]]
            y_train = target [ splits[sort][0] ]
            x_val = data [ splits[sort][1]]
            y_val = target [ splits[sort][1] ]
        else:
            i=1
            if procedure == 'electron':
                i=1
            elif procedure == 'jets':
                i=0
            data_df = electrons_jets(i,d)
            data = norm1(data_df.values)
            target = data['target']
            splits = [(train_index, val_index) for train_index, val_index in cv.split(data,target)]
            x_train = data [ splits[sort][0]]
            y_train = target [ splits[sort][0] ]
            x_val = data [ splits[sort][1]]
            y_val = target [ splits[sort][1] ]
            
    return x_train, x_val, y_train, y_val