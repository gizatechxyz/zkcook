from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CustomRFE(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, n_features_to_select, percentage=0.25, verbose=0):
        self.estimator = estimator
        self.percentage = percentage
        self.verbose = verbose
        self.n_features_to_select = n_features_to_select
        self.model = RFE(estimator=self.estimator, 
                         n_features_to_select=self.n_features_to_select, 
                         step=1,
                         verbose=self.verbose)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.ranking = np.argsort(self.model.ranking_)
        return self

    def transform(self, X, y=None):
        n_features_to_select = int(len(self.ranking) * self.percentage)
        top_features = self.ranking[:n_features_to_select]
        return X[:, top_features]

    def set_params(self, **params):
        self.percentage = params.get('percentage', self.percentage)
        return self
    
    def set_percentage(self, percentage):
        self.percentage = percentage