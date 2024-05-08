from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from .custom_transformers.customPCA import CustomPCA
from .custom_transformers.customRFE import CustomRFE
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
        self._precompute_train_transformations(self.X_train)
        self._precompute_eval_transformations(self.X_eval)

    def _precompute_train_transformations(self, X):
        X_scaled_train = self.scaler.transform(self.X_train)
        self.pca = CustomPCA().fit(X_scaled_train)  # Crear y ajustar PCA
        for percentage in self.percentages:
            self.pca.set_percentage(percentage)
            self.transformations[f'pca_{percentage}_train'] = self.pca.transform(X)

        _n_features_to_select = int(X.shape[1] * 0.25)
        self.rfe = CustomRFE(estimator=self.estimator, 
                             n_features_to_select = _n_features_to_select,
                             verbose=self.verbose).fit(X, self.y_train)
        for percentage in self.percentages:
            self.rfe.set_percentage(percentage)
            self.transformations[f'rfe_{percentage}_train'] = self.rfe.transform(X)

    def _precompute_eval_transformations(self, X):
        X_scaled_eval = self.scaler.transform(self.X_eval)
        for percentage in self.percentages:
            self.pca.set_percentage(percentage)
            self.transformations[f'pca_{percentage}_eval'] = self.pca.transform(X_scaled_eval)

        for percentage in self.percentages:
            self.rfe.set_percentage(percentage)
            self.transformations[f'rfe_{percentage}_eval'] = self.rfe.transform(X)

    def get_transformer(self, method='pca', percentage=0.50):
        if method == 'pca':
            self.pca.set_percentage(percentage)
            return Pipeline([
                ('scaler', self.scaler),
                ('transformer', self.pca)
            ])
        elif method == 'rfe':
            self.rfe.set_percentage(percentage)
            return self.rfe
        else:
            raise ValueError("Método de transformación no encontrado")

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


