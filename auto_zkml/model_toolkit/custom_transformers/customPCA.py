from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


class CustomPCA(BaseEstimator, TransformerMixin):
    def __init__(self, percentage=0.50):
        self.percentage = percentage
        self.model = PCA()

    def fit(self, X, y=None):
        self.model.fit(X)
        return self

    def transform(self, X, y=None):
        n_components = int(len(self.model.explained_variance_ratio_) * self.percentage)
        X_transformed = self.model.transform(X)
        return X_transformed[:, :n_components]

    def set_params(self, **params):
        self.percentage = params.get("percentage", self.percentage)
        return self

    def set_percentage(self, percentage):
        self.percentage = percentage
