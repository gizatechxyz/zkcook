import json

def serialize(model, path, json_name):
    model.save_model(path + json_name)