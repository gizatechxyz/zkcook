import json
import os


def serialize(model, output_path):
    """
    Serializes a model's structure to a JSON file for further use or analysis.

    This function saves the model to a temporary text file, extracts tree information
    from the text format, and transforms it into a structured JSON format that includes
    the base weights, split conditions, child nodes, and split features for each tree.
    The JSON file is saved to the specified path with the given json_name. The temporary
    model file is deleted after the JSON is created.

    Args:
        model (model object): The trained model that is to be serialized.
        output_path (str): The directory path where the JSON file will be saved.

    Raises:
        FileNotFoundError: If the temporary model file could not be found or opened.
        IOError: If there are issues writing the JSON file or deleting the temporary file.
    """
    model.booster_.save_model("./model_tmp.txt")
    with open("./model_tmp.txt") as file:
        model_text = file.read()

    tree_blocks = model_text.split("Tree=")[1:]
    trees = []

    for block in tree_blocks:
        tree_info = {}
        lines = block.strip().split("\n")

        for line in lines:
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key in [
                    "leaf_value",
                    "threshold",
                    "left_child",
                    "right_child",
                    "split_feature",
                ]:
                    values = value.split()
                    if "." in value:
                        tree_info[key] = list(map(float, values))
                    else:
                        tree_info[key] = list(map(int, values))

        trees.append(
            {
                "base_weights": tree_info.get("leaf_value", []),
                "split_conditions": tree_info.get("threshold", []),
                "left_children": tree_info.get("left_child", []),
                "right_children": tree_info.get("right_child", []),
                "split_indices": tree_info.get("split_feature", []),
            }
        )

    json_transformed = {
        "base_score": 0,
        "opt_type": 1,  # TODO: review this value
        "trees_number": len(trees),
        "trees": trees,
    }
    with open(output_path, "w") as f:
        json.dump(json_transformed, f)
    os.remove("./model_tmp.txt")
