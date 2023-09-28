import sys
import os
from argparse import ArgumentParser
from itertools import product


import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
np.random.seed(43)

from prepdata.preparing_data import *
from models import *
import workspace.trainings.autoencoder.classification.prepdata.discriminant_methods as dm

# 0 - 4
ets = [1, 2, 3]
etas = [1, 2, 3, 4]

# Leblon et 4 eta 0

for iet, ieta in product(ets, etas):
    print('!!!!!!!!!!!!!!!!! STARTING ET', iet,'eta',ieta,'TRAINING !!!!!!!!!!!!!!!!!')
    
    print('####################### STARTING PREPARING DATA #######################')
    print('---------------loading data...---------------')
    # '/mnt/cern_data/others/files/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97_et4_eta0.npz'
    # '/mnt/cern_data/others/files/mc16_13TeV.sgn.MC.gammajet.bkg.vetoMC.dijet/mc16_13TeV.sgn.MC.gammajet.bkg.vetoMC.dijet_et4_eta0.npz
    #DATAMC16_PATH = f'/home/marina.juca/data/mc16_13TeV.302236_309995_341330.sgn.boosted_probes.WZ_llqq_plus_radion_ZZ_llqq_plus_ggH3000.merge.25bins.v2_et{iet}_eta{ieta}.npz'
    #DATA17_PATH = f'/home/marina.juca/data/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97_et{iet}_eta{ieta}.npz'

    # singularity run --bind /mnt/cern_data:/home/marina.juca/cern_data ringer_base.sif
    DATA17_PATH = f'/home/marina.juca/cern_data/others/files/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97_et{iet}_eta{ieta}.npz'
    DATAMC16_PATH = f'/home/marina.juca/cern_data/data17_13TeV/files/mc16_13TeV.302236_309995_341330.sgn.boosted_probes.WZ_llqq_plus_radion_ZZ_llqq_plus_ggH3000.merge.25bins.v2/mc16_13TeV.302236_309995_341330.sgn.boosted_probes.WZ_llqq_plus_radion_ZZ_llqq_plus_ggH3000.merge.25bins.v2_et{iet}_eta{ieta}.npz'

    print('loading data17...')
    d17 = loadfile(DATA17_PATH)
    print('done loading data17...')
    print('loading mc16...')
    mc16 = loadfile(DATAMC16_PATH)
    print('---------------done loading data---------------')

    print('---------------normalizing data...---------------')
    data17 = norm1(d17['data'][:, 1:101])
    print('done normalizing data17...')
    data16 = norm1(mc16['data_float'][:, 12:112])
    print('done normalizing mc16...')

    print('getting targets...')
    d17target = d17['target'] # targets from data17 determinate electrons from jets
    mc16target = mc16['target'] # all targets from mc16 are boosted electrons


    normdf17 = pd.DataFrame(data17)
    normdf17['targets'] = d17target


    print('removing jets from data17...')
    criterea = normdf17['targets'] == 0 # creates a boolean array with all jets data from data17
    #normdf17 = normdf17[~criterea] # removes all jets data from data17
    data17 = data17[~criterea] # removes all jets data from data17
    d17target = d17target[~criterea] # removes all jets data from data17


    # criterea = d17target['targets'] == 0 # creates a boolean array with all jets data from data17
    # data17 = data17[~criterea] # removes all jets data from data17
    # d17target = d17target[~criterea] # removes all jets data from data17

    print('assigning 0 to electrons non boosted')
    d17target[d17target == 1] = 0 # all targets of d17 as non boosted
    #mc16target[mc16target != 1] = 0  # it can't be negative because the classifier is by softmax

    print('--------------- creating folds columns...---------------')
    df17 = splits(data17, d17target)
    df16 = splits(data16, mc16target)

    print('---------------done creating folds ---------------')

    print('---------------concatenating dataframes...---------------')
    df = pd.concat([df17,df16])

    for fold in range(10):
        print('############# Fold ', fold, ' #############')
        print('---------------spliting concatenated data into train and val...---------------')
        x_train, y_train, x_val, y_val = train_test(df, fold)
        print('---------------done spliting concatenated data into train and val---------------')
        

        print('---------------creating models...---------------')
        model = DiscriminativeAutoEncoder(
            input_shape=100, layers=[100, 64, 32, 16], classes_num=2, file_path=f'./spits/trainings/et{iet}eta{ieta}/fold{fold}')
        print('compiling models...')
        model.compile()
        print('training models...')
        model.fit(x_train, y_train, validation_data = (x_val, y_val)) #fits with the concatenated data
        print('---------------done training models---------------')

        #model.predict(x_train)   #predicts concatenated data
        print('---------------saving models...---------------')
        model.save_weights(model_name='all_model')
        model.save(model_name='all_model')
        print('---------------done saving models---------------')
        
        print('############# DONE Fold ', fold, ' #############')
    
    print('!!!!!!!!!!!!!!!!! DONE ET', iet,'eta',ieta,'TRAINING !!!!!!!!!!!!!!!!!')

print('####################### DONE TRAINING #######################')
