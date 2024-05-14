from skopt.space import Categorical, Integer, Real


class FeatureSpaceConstants:
    """
    A class for defining and manipulating the feature space for hyperparameter optimization.

    Attributes:
        CONSTANTS (list): Constant parameters used in all model feature spaces.
        MAX_TREES_PER_DEPTH (dict): Specifies the maximum number of trees per depth.
        ADJUST_MIN_PARAMS (list): List of parameters which have minimum values that can be adjusted based on the model's settings.
        FEATURE_SPACES (dict): Different sets of hyperparameters for different types of models.
    """

    CONSTANTS = [
        Categorical([0.25, 0.50, 0.75], name="features_percentage_to_use"),
        Categorical(["pca", "rfe"], name="dimensionality_reduction"),
    ]

    MAX_TREES_PER_DEPTH = {3: 200, 4: 150, 5: 75, 6: 60, 7: 35, 8: 30, 9: 90}

    ADJUST_MIN_PARAMS = ["learning_rate", "rsm"]

    FEATURE_SPACES = {
        "lightgbm": [
            Integer(3, 9, name="max_depth"),
            Integer(20, 40, name="num_leaves"),
            Integer(20, 100, name="min_data_in_leaf"),
            Real(0.1, 0.5, name="feature_fraction"),
            Real(0.1, 0.5, name="bagging_fraction"),
        ],
        "catboost": [
            Integer(3, 9, name="depth"),
            Real(0.1, 0.5, name="rsm"),
            Integer(1, 50, name="min_data_in_leaf"),
            Integer(1, 10, name="leaf_estimation_iterations"),
        ],
        "xgboost": [
            Integer(3, 9, name="max_depth"),
            Real(0, 5, name="gamma"),
            Real(0.1, 0.5, name="subsample"),
            Real(0.1, 0.5, name="colsample_bytree"),
            Real(0.1, 0.5, name="colsample_bylevel"),
        ],
    }

    @staticmethod
    def get_feature_space(model_type, transform_features):
        """
        Retrieves the feature space for the specified model type, optionally enhancing it
        with additional constants if feature transformation is allowed.

        Args:
            model_type (str): Type of the model ('lightgbm', 'catboost', 'xgboost').
            transform_features (bool): Whether additional transformations
                can be applied to the feature space.

        Returns:
            list: A list of hyperparameter spaces configured for the model.
        """
        model_feature_space = FeatureSpaceConstants.FEATURE_SPACES.get(model_type, [])
        if transform_features:
            model_feature_space = model_feature_space + FeatureSpaceConstants.CONSTANTS
        return model_feature_space

    @staticmethod
    def adjust_dimension(dimension, model_param_value):
        """
        Adjusts the boundaries of a given hyperparameter based on the current model's parameter value.

        Args:
            dimension (Dimension): The skopt dimension object representing a hyperparameter.
            model_param_value (float): The current value of the parameter in the model.

        Returns:
            Dimension: The adjusted hyperparameter dimension with updated boundaries.
        """
        if dimension.name in FeatureSpaceConstants.ADJUST_MIN_PARAMS and model_param_value < dimension.high:
            print(f"Adjusting {dimension.name} to new range: [{model_param_value}, {dimension.high}]")
            return Real(model_param_value, dimension.high, name=dimension.name)
        elif dimension.name not in FeatureSpaceConstants.ADJUST_MIN_PARAMS:
            if model_param_value <= dimension.low:
                print(f"Adjusting {dimension.name} to new range: [{model_param_value}, {dimension.high}]")
                return type(dimension)(model_param_value, dimension.high, name=dimension.name)
            elif model_param_value < dimension.high:
                print(f"Adjusting {dimension.name} to new range: [{dimension.low}, {model_param_value}]")
                return type(dimension)(dimension.low, model_param_value, name=dimension.name)
        return dimension

    @staticmethod
    def adjust_search_space(space, model_params):
        """
        Adjusts the entire search space based on the model's parameters, potentially altering
        each dimension to fit better with current model configurations.

        Args:
            space (list of Dimension): The initial search space dimensions.
            model_params (dict): Current parameter values from the model.

        Returns:
            list: The list of adjusted dimensions forming the new search space.
        """
        adjusted_space = []
        for dimension in space:
            if dimension.name in model_params:
                adjusted_dimension = FeatureSpaceConstants.adjust_dimension(dimension, model_params[dimension.name])
                if adjusted_dimension:
                    adjusted_space.append(adjusted_dimension)
            else:
                adjusted_space.append(dimension)
        return adjusted_space
