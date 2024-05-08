import json
import os

def serialize(model, path, json_name):
    model.booster_.save_model(path + "/model_tmp.txt")
    with open(path + "/model_tmp.txt", 'r') as file:
        model_text = file.read()

    tree_blocks = model_text.split("Tree=")[1:]
    trees = []

    for block in tree_blocks:
        tree_info = {}
        lines = block.strip().split('\n')
        
        for line in lines:
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                if key in ["leaf_value", "threshold", "left_child", "right_child", "split_feature"]:
                    values = value.split()
                    if '.' in value:
                        tree_info[key] = list(map(float, values))
                    else:
                        tree_info[key] = list(map(int, values))

        trees.append({
            "base_weights": tree_info.get("leaf_value", []),
            "split_conditions": tree_info.get("threshold", []),
            "left_children": tree_info.get("left_child", []),
            "right_children": tree_info.get("right_child", []),
            "split_indices": tree_info.get("split_feature", [])
        })

    json_transformed = {
        "base_score": 0,
        "opt_type": 1, #TODO: review this value
        "trees_number": len(trees),
        "trees": trees
    }
    with open(path + json_name, 'w') as f:
        json.dump(json_transformed, f)
    os.remove(path + "/model_tmp.txt")