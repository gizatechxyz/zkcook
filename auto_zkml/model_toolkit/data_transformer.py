from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from auto_zkml.model_toolkit.custom_transformers.customPCA import CustomPCA
from auto_zkml.model_toolkit.custom_transformers.customRFE import CustomRFE


class DataTransformer(BaseEstimator, TransformerMixin):
    """
    A class for transforming data using PCA and RFE as part of data preprocessing steps for machine learning models.

    Attributes:
        estimator (estimator object): The machine learning estimator for RFE.
        X_train (DataFrame): Training feature dataset.
        y_train (Series): Training target dataset.
        X_eval (DataFrame): Evaluation feature dataset.
        percentages (list): List of percentages defining the extent of dimension reduction.
        verbose (int): Verbosity mode.
        scaler (StandardScaler): Instance of StandardScaler, fitted to X_train.
        transformations (dict): Dictionary storing transformation matrices for different percentages and methods.
    """

    def __init__(self, estimator, X_train, y_train, X_eval, percentages, verbose=0):
        """
        Initializes the DataTransformer with training and evaluation data, percentages for transformation, and an estimator for RFE.

        Parameters:
            estimator: The machine learning estimator compatible with RFE.
            X_train (array-like): Training data features.
            y_train (array-like): Training data labels.
            X_eval (array-like): Evaluation data features.
            percentages (list): Percentages to determine the extent of dimensionality reduction.
            verbose (int): Controls the verbosity of the process.
        """
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
        """
        Precomputes transformations for both training and evaluation datasets and stores them in the transformations dictionary.
        """
        self._precompute_train_transformations(self.X_train)
        self._precompute_eval_transformations(self.X_eval)

    def _precompute_train_transformations(self, X):
        """
        Precomputes and stores PCA and RFE transformations for the training dataset based on defined percentages.

        Parameters:
            X (DataFrame): The training dataset to compute transformations for.
        """
        X_scaled_train = self.scaler.transform(self.X_train)
        self.pca = CustomPCA().fit(X_scaled_train)
        for percentage in self.percentages:
            self.pca.set_percentage(percentage)
            self.transformations[f"pca_{percentage}_train"] = self.pca.transform(X)
        _n_features_to_select = int(X.shape[1] * 0.25)
        self.rfe = CustomRFE(
            estimator=self.estimator,
            n_features_to_select=_n_features_to_select,
            verbose=self.verbose,
        ).fit(X, self.y_train)
        for percentage in self.percentages:
            self.rfe.set_percentage(percentage)
            self.transformations[f"rfe_{percentage}_train"] = self.rfe.transform(X)

    def _precompute_eval_transformations(self, X):
        """
        Precomputes and stores PCA and RFE transformations for the evaluation dataset based on defined percentages.

        Parameters:
            X (DataFrame): The evaluation dataset to compute transformations for.
        """
        X_scaled_eval = self.scaler.transform(self.X_eval)
        for percentage in self.percentages:
            self.pca.set_percentage(percentage)
            self.transformations[f"pca_{percentage}_eval"] = self.pca.transform(X_scaled_eval)

        for percentage in self.percentages:
            self.rfe.set_percentage(percentage)
            self.transformations[f"rfe_{percentage}_eval"] = self.rfe.transform(X)

    def get_transformer(self, method="pca", percentage=0.50):
        """
        Retrieves a transformer pipeline based on the specified method and percentage.

        Parameters:
            method (str): The method of transformation ('pca' or 'rfe').
            percentage (float): The percentage of features to retain.

        Returns:
            Pipeline: A scikit-learn pipeline object with scaling and the specified transformation.
        """

        if method == "pca":
            self.pca.set_percentage(percentage)
            return Pipeline([("scaler", self.scaler), ("transformer", self.pca)])
        elif method == "rfe":
            self.rfe.set_percentage(percentage)
            return self.rfe
        else:
            raise ValueError("Método de transformación no encontrado")

    def apply_transformations(self, X_eval):
        """
        Applies precomputed transformations to the evaluation dataset.

        Parameters:
            X_eval (DataFrame): The evaluation data to apply transformations to.

        Returns:
            dict: A dictionary of transformed datasets for each method and percentage combination.
        """
        eval_transformations = {}
        for method, percentage in self._get_method_percentage_combinations():
            key = f"{method}_{percentage}_eval"
            if key in self.transformations:
                eval_transformations[key] = self.transformations[key]
            else:
                raise ValueError(f"No se encontró una transformación precalculada para {key}")
        return eval_transformations

    def _get_method_percentage_combinations(self):
        """
        Generates all possible combinations of transformation methods and percentages.

        Yields:
            tuple: Each combination of method and percentage.
        """
        for method in ["pca", "rfe"]:
            for percentage in self.percentages:
                yield method, percentage

    def get_transformation(self, method="pca", percentage=0.50, dataset_type="train"):
        """
        Retrieves a specific transformation matrix from the transformations dictionary based on method, percentage, and dataset type.

        Parameters:
            method (str): The transformation method ('pca' or 'rfe').
            percentage (float): The percentage of features to retain.
            dataset_type (str): The dataset type ('train' or 'eval').

        Returns:
            ndarray: The transformation matrix for the specified parameters.
        """
        key = f"{method}_{percentage}_{dataset_type}"
        if key in self.transformations:
            return self.transformations[key]
        else:
            raise ValueError(f"No se encontró una transformación precalculada para {key}")
