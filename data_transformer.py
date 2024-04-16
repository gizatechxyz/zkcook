from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import numpy as np


class DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, X_train, y_train, X_eval, percentages, verbose=0):
        self.estimator = estimator
        self.X_train = X_train
        self.y_train = y_train
        self.X_eval = X_eval
        self.percentages = percentages
        self.verbose = verbose
        self.scaler = StandardScaler().fit(X_train)
        self.transformations = {}
        self.precompute_transformations()

    def precompute_transformations(self):

        X_scaled_train = self.scaler.transform(self.X_train)
        X_scaled_eval = self.scaler.transform(self.X_eval)
        self._precompute_train_transformations(X_scaled_train)
        self._precompute_eval_transformations(X_scaled_eval)

    def _precompute_train_transformations(self, X):
        self.pca = PCA().fit(X)
        for percentage in self.percentages:
            n_components = int(len(self.pca.explained_variance_ratio_) * percentage)
            transformed_data = self.pca.transform(X)[:, :n_components]
            self.transformations[f'pca_{percentage}_train'] = transformed_data

        self.rfe = RFE(estimator=self.estimator, n_features_to_select=None, step=1, verbose=self.verbose)
        self.rfe.fit(X, self.y_train)
        support_ = self.rfe.get_support(indices=True)
        for percentage in self.percentages:
            n_features_to_select = int(len(support_) * percentage)
            selected_features = support_[:n_features_to_select]
            self.transformations[f'rfe_{percentage}_train'] = X[:, selected_features]

    def _precompute_eval_transformations(self, X):
        for percentage in self.percentages:
            n_components = int(len(self.pca.explained_variance_ratio_) * percentage)
            transformed_data = self.pca.transform(X)[:, :n_components]
            self.transformations[f'pca_{percentage}_eval'] = transformed_data
        support_ = self.rfe.get_support(indices=True)
        for percentage in self.percentages:
            n_features_to_select = int(len(support_) * percentage)
            selected_features = support_[:n_features_to_select]
            self.transformations[f'rfe_{percentage}_eval'] = X[:, selected_features]

    def apply_transformations(self, X_eval):
        # Apply precalculated transformations to the evaluation dataset
        X_scaled_eval = self.scaler.transform(X_eval)
        eval_transformations = {}
        for method, percentage in self._get_method_percentage_combinations():
            key = f'{method}_{percentage}_eval'
            if key in self.transformations:
                eval_transformations[key] = self.transformations[key]
            else:
                raise ValueError(f"No se encontró una transformación precalculada para {key}")
        return eval_transformations

    def _get_method_percentage_combinations(self):
        # Generate all combinations of method and percentage
        for method in ['pca', 'rfe']:
            for percentage in self.percentages:
                yield method, percentage
    
    def get_transformation(self, method='pca', percentage=0.50, dataset_type='train'):

        key = f'{method}_{percentage}_{dataset_type}'
        if key in self.transformations:
            return self.transformations[key]
        else:
            raise ValueError(f"No se encontró una transformación precalculada para {key}")


