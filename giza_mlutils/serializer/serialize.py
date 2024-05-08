from giza_mlutils.model_toolkit.model_info import ModelParameterExtractor
from giza_mlutils.serializer import xg, lgbm

def serialize_model(model, path, json_name):
    extractor = ModelParameterExtractor()
    _, model_class = extractor.get_model_params_and_class(model)
    model_type = ModelParameterExtractor.get_package_name(model_class)
    if model_type == "xgboost":
        xg.serialize(model, path, json_name)
    else:
        lgbm.serialize(model, path, json_name)