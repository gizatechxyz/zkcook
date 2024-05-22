import json


class ModelParameterExtractor:
    def __init__(self):
        self.model_extractors_by_name = {
            "XGBRegressor": self.extract_params_from_xgb,
            "XGBClassifier": self.extract_params_from_xgb,
            "LGBMClassifier": self.extract_params_from_lgb,
            "LGBMRegressor": self.extract_params_from_lgb,
            "Booster": self.extract_params_from_xgb,
        }

    def extract_params_from_xgb(self, model):
        if hasattr(model, "get_booster"):
            config = model.get_booster().save_config()
            return self._transform_xgb_params(config)
        else:
            config = model.save_config()
            return self._transform_xgb_params(config)

    def extract_params_from_lgb(self, model):
        params = model.get_params(deep=True)
        if params["max_depth"] == -1:
            params["max_depth"] = 9
        return params

    def extract_params_from_catboost(self, model):
        return model.get_all_params()

    def _transform_xgb_params(self, raw_xgb_params):
        def convert_value(value):
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return value

        model_params = json.loads(raw_xgb_params)
        model_params = model_params["learner"]["gradient_booster"]["tree_train_param"]
        converted_params = {}
        for param, value in model_params.items():
            converted_params[param] = convert_value(value)
        return converted_params

    def get_model_params_and_class(self, input_model):
        model_class_name = input_model.__class__.__name__
        if model_class_name == "Booster":
            model_class_name = type(input_model).__name__

        if model_class_name in self.model_extractors_by_name:
            extractor_func = self.model_extractors_by_name[model_class_name]
            params = extractor_func(input_model)
            return params, input_model.__class__
        else:
            raise ValueError(f"No extractor found for model type: {model_class_name}.")

    def get_package_name(model_class):
        return model_class.__module__.split(".")[0]
