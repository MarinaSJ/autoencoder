from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import euclidean
import numpy as np


def move_to_means(train_data, train_classes, a=0.2):
    n_classes = np.max(train_classes)+1
    print ('------------------- n_classes : ', n_classes, ' a : ', a, ' -----------------------')
    centers = np.zeros((n_classes, train_data.shape[1]))
    new_train_data = np.zeros(train_data.shape)
    for i in range(n_classes):
        indices = np.squeeze(np.where(train_classes==i))
        centers[i,...] = np.mean(train_data[indices,...], axis=0)
    for i in range(train_data.shape[0]):
        new_train_data[i,...] = (1-a)*train_data[i,...] + a*centers[train_classes[i],...]
    return new_train_data


def move_away_from_means(train_data, train_classes, a=0.1, k=3):
    n_classes = np.max(train_classes)+1
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


def move_to_neighbors(train_data, train_classes, k=3, a=0.2):
    n_classes = np.max(train_classes) + 1
    new_train_data = np.zeros(train_data.shape)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1)
    for i in range(n_classes):
        ind = np.squeeze(np.where(train_classes == i))
        nbrs.fit(train_data[ind, ...])
        n_dis, n_ind = nbrs.kneighbors(train_data[ind, ...], k, return_distance=True)
        for j in range(train_data[ind,...].shape[0]):
            neigh = train_data[ind,...][n_ind[j,...], ...][1:, ...]
            new_train_data[ind[j],...] = (1-a)*train_data[ind[j],...] + a*np.mean(neigh, axis=0)
    return new_train_data


def move_to_safer_neighbors(train_data, train_classes, k=3, a=0.2):
    n_classes = np.max(train_classes) + 1
    centers = np.zeros((n_classes, train_data.shape[1]))
    for j in range(n_classes):
        indices = np.squeeze(np.where(train_classes == j))
        centers[j, ...] = np.mean(train_data[indices, ...], axis=0)
    new_train_data = np.zeros(train_data.shape)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1)
    for i in range(n_classes):
        ind = np.squeeze(np.where(train_classes == i))
        nbrs.fit(train_data[ind, ...])
        n_ind = nbrs.kneighbors(train_data[ind, ...], k, return_distance=False)
        for j in range(train_data[ind,...].shape[0]):
            neigh = train_data[ind,...][n_ind[j,...], ...]
            s_neigh = []
            s = 0
            for n in range(k):
                if euclidean(neigh[n,...], centers[train_classes[j,...],...]) <= euclidean(train_data[j,...], centers[train_classes[j,...],...]):
                    s_neigh.append(neigh[n,...])
                    s = s+1
            if s>0:
                new_train_data[ind[j],...] = (1-a)*train_data[ind[j],...] + a*np.mean(np.squeeze(np.asarray(s_neigh)), axis=0)
            else:
                new_train_data[ind[j], ...] = train_data[ind[j], ...]
    return new_train_data


def move_away_from_rivals(train_data, train_classes, k=3, a=0.2):
    n_classes = np.max(train_classes) + 1
    new_train_data = train_data
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1)
    # nbrs = LSHForest(random_state=43, n_neighbors=k, n_estimators=2)
    for i in range(n_classes):
        ind = np.squeeze(np.where(train_classes != i))
        sc_ind = np.squeeze(np.where(train_classes == i))
        nbrs.fit(train_data[ind,...])
        if sc_ind.size >1:
            n_ind = nbrs.kneighbors(train_data[sc_ind,...], k, return_distance=False)
            for j in range(train_data[sc_ind,...].shape[0]):
                if sc_ind.size>1:
                    neigh = train_data[ind,...][n_ind[j,...],...]
                    new_train_data[sc_ind[j],...] = (1+a)*train_data[sc_ind[j],...] - a*np.squeeze(np.mean(neigh, axis=0))
    return new_train_data

