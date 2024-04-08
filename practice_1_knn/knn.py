import numpy as np


class KNN:
    """
    K-neariest-neighbor classifier using L1 ('manhattan') or L2 ('euclidean') metric
    """
    def __init__(self, k=1, metric=None):
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict classes for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool_:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        if self.metric == 'manhattan':
            for i_test in range(num_test):
                for i_train in range(num_train):
                    dists[i_test][i_train] = np.sum(np.abs(X[i_test] - self.train_X[i_train]))
        elif self.metric == 'euclidean':
            for i_test in range(num_test):
                for i_train in range(num_train):
                    dists[i_test][i_train] = np.sum((X[i_test] - self.train_X[i_train])**2)
        return dists 

    def compute_distances_one_loop(self, X):
        '''
        Computes distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        if self.metric == 'manhattan':
            for i_test in range(num_test):
                dists[i_test] = np.sum(np.abs(X[i_test] - self.train_X), axis=1)
        elif self.metric == 'euclidean':
            for i_test in range(num_test):
                dists[i_test] = np.sum((X[i_test] - self.train_X)**2, axis=1)
        return dists

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        # Using float32 to to save memory - the default is float64
        
        # we want something like this...
        # X -> (num_test, 1, num_features)
        # train_X -> (num_train, num_features)
        dists = np.zeros((num_test, num_train), np.float32)
        if self.metric == 'manhattan':
            dists = np.sum(np.abs(X[:, np.newaxis] - self.train_X), axis=2)
        if self.metric == 'euclidean':
            dists = np.sum((X[:, np.newaxis] - self.train_X)**2, axis=2)
        
        return dists 

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool_)
        for i in range(num_test):
            nearest_neighbors = np.argsort(dists[i])[:self.k]
            most_common = np.bincount(self.train_y[nearest_neighbors]).argmax()
            pred[i] = most_common
        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        num_test = dists.shape[0]
        num_test = dists.shape[0]
        pred = np.zeros(num_test, int)
        for i in range(num_test):
            nearest_neighbors = np.argsort(dists[i])[:self.k]
            pred[i] = np.argmax(np.bincount(self.train_y[nearest_neighbors]))
        return pred
