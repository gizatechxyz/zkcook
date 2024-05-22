import json


def serialize(model, output_path):
    booster = model.get_booster()
    model_bytes = booster.save_raw(raw_format="json")
    model_json_str = model_bytes.decode("utf-8")
    model_json = json.loads(model_json_str)
    opt_type = model_json["learner"]["objective"]["name"].lower()

    if "binary" in opt_type:
        opt_type = 1
    elif "reg" in opt_type:
        opt_type = 0
    else:
        raise ValueError("The model should be a classifier or regressor model.")

    new_fields = {"model_type": "xgboost", "opt_type": opt_type}
    combined_json = {**new_fields, **model_json}

    combined_json_str = json.dumps(combined_json)

    with open(output_path, "w") as file:
        file.write(combined_json_str)
