from auto_zkml.model_toolkit.model_info import ModelParameterExtractor
from auto_zkml.serializer import lgbm, xg


def serialize_model(model, output_path):
    """
    Serializes a model to a JSON file depending on the type of model provided (e.g., XGBoost or LightGBM).

    This function uses the ModelParameterExtractor to determine the type of the model, and based
    on the type, it delegates the serialization task to the appropriate serializer (xg or lgbm).
    It saves the serialized model in the specified JSON format under the given directory and filename.

    Args:
        model (object): The model instance that needs to be serialized.
        output_path (str): The directory path where the JSON file will be saved.

    Raises:
        ValueError: If the model type is unrecognized or if any issue occurs during the serialization process.
    """
    extractor = ModelParameterExtractor()
    _, model_class = extractor.get_model_params_and_class(model)
    model_type = ModelParameterExtractor.get_package_name(model_class)
    if model_type == "xgboost":
        xg.serialize(model, output_path)
    else:
        lgbm.serialize(model, output_path)
