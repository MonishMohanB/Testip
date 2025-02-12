import json
import yaml
from typing import Union, Dict, Any

def convert_to_dict(input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Converts YAML, JSON, or dict input into a Python dictionary.

    Args:
        input_data (Union[str, Dict[str, Any]]): Input data in YAML, JSON, or dict format.

    Returns:
        Dict[str, Any]: The converted dictionary.

    Raises:
        ValueError: If the input data is not in a valid format.
    """
    if isinstance(input_data, dict):
        # If the input is already a dictionary, return it directly
        return input_data

    if not isinstance(input_data, str):
        raise ValueError("Input must be a string (YAML/JSON) or a dictionary.")

    try:
        # Try parsing as JSON
        return json.loads(input_data)
    except json.JSONDecodeError:
        try:
            # Try parsing as YAML
            return yaml.safe_load(input_data)
        except yaml.YAMLError:
            # If neither JSON nor YAML parsing works, raise an error
            raise ValueError("Input is not valid JSON or YAML.")

# Example Usage
if __name__ == "__main__":
    # Test cases
    json_input = '{"name": "John", "age": 30}'
    yaml_input = """
    name: John
    age: 30
    """
    dict_input = {"name": "John", "age": 30}
    invalid_input = "This is not valid JSON or YAML."

    # Test JSON input
    try:
        print("JSON Input:", convert_to_dict(json_input))
    except ValueError as e:
        print(f"Error with JSON input: {e}")

    # Test YAML input
    try:
        print("YAML Input:", convert_to_dict(yaml_input))
    except ValueError as e:
        print(f"Error with YAML input: {e}")

    # Test dict input
    try:
        print("Dict Input:", convert_to_dict(dict_input))
    except ValueError as e:
        print(f"Error with dict input: {e}")

    # Test invalid input
    try:
        print("Invalid Input:", convert_to_dict(invalid_input))
    except ValueError as e:
        print(f"Error with invalid input: {e}")
