from skopt.utils import use_named_args
from skopt import gp_minimize

from .model_toolkit.model_info import ModelParameterExtractor
from .model_toolkit.feature_models_space import FeatureSpaceConstants
from .model_toolkit.data_transformer import DataTransformer
from .model_toolkit.model_trainer import ModelTrainer
from .model_toolkit.model_evaluator import ModelEvaluator
from .model_toolkit.metrics import check_metric_optimization

def mcr(model,X_train, y_train, X_eval, y_eval, eval_metric, can_we_transform_your_features):
    extractor = ModelParameterExtractor()
    model_params, model_class = extractor.get_model_params_and_class(model) #TODO: revisar
    model_type = ModelParameterExtractor.get_package_name(model_class)
    metric = check_metric_optimization(model_type, eval_metric)
    initial_feature_space = FeatureSpaceConstants.get_feature_space(model_type, can_we_transform_your_features)
    adjusted_feature_space = FeatureSpaceConstants.adjust_search_space(initial_feature_space, model_params)
    if can_we_transform_your_features:
        features_percentage_to_use_list = list([cat for cat in FeatureSpaceConstants.CONSTANTS 
                                           if cat.name == 'features_percentage_to_use'][0].categories)
        data_transformer = DataTransformer(estimator=model_class(), 
                                           X_train=X_train, 
                                           y_train=y_train, 
                                           X_eval = X_eval,
                                           percentages=features_percentage_to_use_list, 
                                           verbose=0)
    model_trainer = ModelTrainer(model_class, model_type)
    model_evaluator = ModelEvaluator(model_type, eval_metric)
    @use_named_args(adjusted_feature_space)
    def _objective(**kwargs):
        if can_we_transform_your_features:
            features_percentage_to_use = kwargs.pop("features_percentage_to_use")
            red_method = kwargs.pop('dimensionality_reduction')
            if red_method == 'pca':
                X_train_aux = data_transformer.get_transformation(method='pca', percentage=features_percentage_to_use, dataset_type='train')
                X_eval_aux = data_transformer.get_transformation(method='pca', percentage=features_percentage_to_use, dataset_type='eval')
            elif red_method == 'rfe':
                X_train_aux = data_transformer.get_transformation(method='rfe', percentage=features_percentage_to_use, dataset_type='train')
                X_eval_aux = data_transformer.get_transformation(method='rfe', percentage=features_percentage_to_use, dataset_type='eval')
            else:
                X_train_aux = X_train
                X_eval_aux = X_eval
        else:
            X_train_aux = X_train
            X_eval_aux = X_eval

        kwargs["n_estimators"] = FeatureSpaceConstants.MAX_TREES_PER_DEPTH[kwargs["max_depth"]]
        model_parameters = {**model_params, **kwargs}
        retrained_model = model_trainer.train_model( 
                                          X_train_aux, 
                                          y_train, 
                                          X_eval_aux, 
                                          y_eval, 
                                          model_parameters, 
                                          eval_metric= eval_metric,
                                          early_stopping_rounds=10)

        final_eval = model_evaluator.evaluate_model(retrained_model)
        if metric == "maximize":
            final_eval = -final_eval
        return final_eval
    
    res = gp_minimize(_objective, adjusted_feature_space, n_calls=10, random_state=0)

    best_params_dict = {dim.name: value for dim, value in zip(res.space, res.x)}
    if can_we_transform_your_features:
        dim_red = best_params_dict.pop("dimensionality_reduction")
        perc_to_use = best_params_dict.pop("features_percentage_to_use")
    model_params = {**model_params, **best_params_dict}
    best_params_dict["n_estimators"] = FeatureSpaceConstants.MAX_TREES_PER_DEPTH[model_params["max_depth"]]

    if can_we_transform_your_features:
        transformer = data_transformer.get_transformer(method = dim_red,  percentage = perc_to_use)
        X_train = transformer.transform(X_train)
        X_eval = transformer.transform(X_eval)
    else:
        transformer = None

    retrained_model = model_trainer.train_model( 
                                        X_train, 
                                        y_train, 
                                        X_eval, 
                                        y_eval, 
                                        best_params_dict, 
                                        eval_metric= eval_metric,
                                        early_stopping_rounds=10)
    
    return retrained_model, transformer