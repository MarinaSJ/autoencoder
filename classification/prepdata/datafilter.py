'''

This model filters the data to be used in the training of the autoencoder to only include electrons.
This way the autoencoder will only learn to reconstruct electrons and will filter out other signals,
presenting a more filtered input to the classifier.

'''

import pandas as pd


def electrons_jets(i,d):
    '''
    
    Filters the data to only include electrons.
    
    '''
    
    #data = d['data']
    # new df
    data_df = pd.DataFrame(data=d['data'], columns=d['features'])
    data_df['target'] = d['target']
    
    filtered_df = data_df.loc[data_df['target']==i] #i=1 for electrons, i=0 for jets
    
    return filtered_df




def rings_reduction(n,d):
    '''

    This model is part of the preparing the data
    It is used to reduce the number of rings in each layer of the calorimeter, divinding them by 2, 4 or 8.

    '''
    # new df
    data_df = pd.DataFrame(data=d['data'], columns=d['features'])
    
    # norm considering all rings
    #all_rings = ['L2Calo_ring_%i' %iring for iring in range(100)]
    #data_df.loc[:, all_rings] = norm1(data_df[all_rings].values)

    # for new training, we selected 1/4 of rings in each layer
    #pre-sample - 8 rings
    # EM1 - 64 rings
    # EM2 - 8 rings
    # EM3 - 8 rings
    # Had1 - 4 rings
    # Had2 - 4 rings
    # Had3 - 4 rings
    prefix = 'L2Calo_ring_%i'

    # rings presmaple 
    presample = [prefix %iring for iring in range(8//n)]

    # EM1 list
    sum_rings = 8
    em1 = [prefix %iring for iring in range(sum_rings, sum_rings+(64//n))]

    # EM2 list
    sum_rings = 8+64
    em2 = [prefix %iring for iring in range(sum_rings, sum_rings+(8//n))]

    # EM3 list
    sum_rings = 8+64+8
    em3 = [prefix %iring for iring in range(sum_rings, sum_rings+(8//n))]

    # HAD1 list
    sum_rings = 8+64+8+8
    had1 = [prefix %iring for iring in range(sum_rings, sum_rings+(4//n))]

    # HAD2 list
    sum_rings = 8+64+8+8+4
    had2 = [prefix %iring for iring in range(sum_rings, sum_rings+(4//n))]

    # HAD3 list
    sum_rings = 8+64+8+8+4+4
    had3 = [prefix %iring for iring in range(sum_rings, sum_rings+(4//n))]

    selection_list = presample+em1+em2+em3+had1+had2+had3
    
    return data_df, selection_list