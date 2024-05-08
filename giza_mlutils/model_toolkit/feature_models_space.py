from skopt.space import Categorical, Real, Integer



class FeatureSpaceConstants:
    CONSTANTS = [
        Categorical([0.25, 0.50, 0.75], name='features_percentage_to_use'),
        Categorical(['pca','rfe'], name='dimensionality_reduction'),
    ]

    MAX_TREES_PER_DEPTH = {3:200, 4:150, 5:75, 6:60, 7:35, 8:30, 9:90 }

    ADJUST_MIN_PARAMS = ['learning_rate','rsm']

    FEATURE_SPACES = {
        'lightgbm': [
            Integer(3, 9, name='max_depth'),
            Integer(20, 40, name='num_leaves'),
            Integer(20, 100, name='min_data_in_leaf'),
            Real(0.1, 0.5, name='feature_fraction'),
            Real(0.1, 0.5, name='bagging_fraction'),
        ],
        'catboost': [
            Integer(3, 9, name='depth'),
            Real(0.1, 0.5, name='rsm'),
            Integer(1, 50, name='min_data_in_leaf'),
            Integer(1, 10, name='leaf_estimation_iterations'),
        ],
        'xgboost': [
            Integer(3, 9, name='max_depth'),
            Real(0, 5, name='gamma'),
            Real(0.1, 0.5, name='subsample'),
            Real(0.1, 0.5, name='colsample_bytree'),
            Real(0.1, 0.5, name='colsample_bylevel'),
        ]
    }

    @staticmethod
    def get_feature_space(model_type, can_we_transform_your_features):
        model_feature_space = FeatureSpaceConstants.FEATURE_SPACES.get(model_type, [])
        if can_we_transform_your_features:
            model_feature_space = model_feature_space + FeatureSpaceConstants.CONSTANTS
        return model_feature_space

    @staticmethod
    def adjust_dimension(dimension, model_param_value):
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
        # Añadir if para la profundidad de lightgbm
        adjusted_space = []
        for dimension in space:
            if dimension.name in model_params:
                adjusted_dimension = FeatureSpaceConstants.adjust_dimension(dimension, model_params[dimension.name])
                if adjusted_dimension:
                    adjusted_space.append(adjusted_dimension)
            else:
                adjusted_space.append(dimension)
        return adjusted_space
