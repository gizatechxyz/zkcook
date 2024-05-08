from giza_mlutils.model_toolkit.model_info import ModelParameterExtractor
from giza_mlutils.serializer import xg, lgbm

def serialize_model(model, path, json_name):
    """
    Serializes a model to a JSON file depending on the type of model provided (e.g., XGBoost or LightGBM).
    
    This function uses the ModelParameterExtractor to determine the type of the model, and based
    on the type, it delegates the serialization task to the appropriate serializer (xg or lgbm).
    It saves the serialized model in the specified JSON format under the given directory and filename.
    
    Args:
        model (object): The model instance that needs to be serialized.
        path (str): The directory path where the JSON file will be saved.
        json_name (str): The filename for the serialized model data in JSON format.

    Raises:
        ValueError: If the model type is unrecognized or if any issue occurs during the serialization process.
    """
    extractor = ModelParameterExtractor()
    _, model_class = extractor.get_model_params_and_class(model)
    model_type = ModelParameterExtractor.get_package_name(model_class)
    if model_type == "xgboost":
        xg.serialize(model, path, json_name)
    else:
        lgbm.serialize(model, path, json_name)