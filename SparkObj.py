import yaml

def to_yaml_without_data(obj, exclude_keys=('training_data', 'validation_data')):
    # Extract all attributes except the ones we want to exclude
    attr_dict = {
        k: v
        for k, v in obj.__dict__.items()
        if k not in exclude_keys
    }

    # Optional: clean up Spark-related non-serializable items if needed

    # Convert to YAML
    return yaml.dump(attr_dict, default_flow_style=False)

# Usage:
yaml_output = to_yaml_without_data(your_class_instance)
with open("output.yaml", "w") as f:
    f.write(yaml_output)

'''
Sure! Here's an improved version that recursively removes Spark DataFrames and other non-serializable types from nested dictionaries, lists, or class objects. It can also handle nested custom objects.

Full '''

import yaml
from pyspark.sql import DataFrame

def is_serializable(value):
    # You can add more checks for non-serializable types here
    return not isinstance(value, DataFrame)

def clean_obj(obj, exclude_keys=('training_data', 'validation_data'), seen=None):
    if seen is None:
        seen = set()

    if id(obj) in seen:
        return None  # Prevent infinite recursion
    seen.add(id(obj))

    if isinstance(obj, dict):
        return {
            k: clean_obj(v, exclude_keys, seen)
            for k, v in obj.items()
            if k not in exclude_keys and is_serializable(v)
        }
    elif isinstance(obj, (list, tuple, set)):
        return type(obj)(
            clean_obj(item, exclude_keys, seen)
            for item in obj
            if is_serializable(item)
        )
    elif hasattr(obj, '__dict__'):
        return clean_obj(obj.__dict__, exclude_keys, seen)
    elif is_serializable(obj):
        return obj
    else:
        return None

def save_clean_yaml(obj, filename, exclude_keys=('training_data', 'validation_data')):
    cleaned = clean_obj(obj, exclude_keys)
    with open(filename, 'w') as f:
        yaml.dump(cleaned, f, default_flow_style=False)

# Usage
# save_clean_yaml(your_class_instance, "clean_output.yaml")
