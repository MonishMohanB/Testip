import json
import yaml
from typing import Union, Dict, Any

def read_and_convert_to_dict(file_path: str = None, input_type: str = None, input_data: Union[Dict[str, Any], str] = None) -> Dict[str, Any]:
    """
    Reads data from a file or input data and converts it to a Python dictionary.

    Args:
        file_path (str, optional): Path to the file containing the data. Defaults to None.
        input_type (str, optional): Type of input data ('json', 'yaml', or 'dict'). Defaults to None.
        input_data (Union[Dict[str, Any], str], optional): Input data as a dictionary or string. Defaults to None.

    Returns:
        Dict[str, Any]: The converted dictionary.

    Raises:
        ValueError: If the input type is invalid or the file cannot be read.
        FileNotFoundError: If the file path is provided but the file does not exist.
    """
    # Validate input arguments
    if file_path is None and input_data is None:
        raise ValueError("Either file_path or input_data must be provided.")
    if file_path is not None and input_data is not None:
        raise ValueError("Only one of file_path or input_data should be provided.")
    if input_type is None:
        raise ValueError("input_type must be provided ('json', 'yaml', or 'dict').")

    # If input_data is provided, use it directly
    if input_data is not None:
        if input_type == "dict":
            if not isinstance(input_data, dict):
                raise ValueError("input_data must be a dictionary for input_type 'dict'.")
            return input_data
        elif input_type in ["json", "yaml"]:
            if not isinstance(input_data, str):
                raise ValueError("input_data must be a string for input_type 'json' or 'yaml'.")
            data_str = input_data
        else:
            raise ValueError("Invalid input_type. Must be 'json', 'yaml', or 'dict'.")
    else:
        # Read data from file
        try:
            with open(file_path, "r") as file:
                data_str = file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        except Exception as e:
            raise ValueError(f"Error reading file: {e}")

    # Convert data to dictionary based on input_type
    try:
        if input_type == "json":
            return json.loads(data_str)
        elif input_type == "yaml":
            return yaml.safe_load(data_str)
        else:
            raise ValueError("Invalid input_type. Must be 'json', 'yaml', or 'dict'.")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format.")
    except yaml.YAMLError:
        raise ValueError("Invalid YAML format.")
    except Exception as e:
        raise ValueError(f"Error converting data to dictionary: {e}")

# Example Usage
if __name__ == "__main__":
    # Test cases
    json_file = "data.json"
    yaml_file = "data.yaml"
    dict_input = {"name": "John", "age": 30}
    json_input = '{"name": "John", "age": 30}'
    yaml_input = """
    name: John
    age: 30
    """

    # Test JSON file
    try:
        print("JSON File:", read_and_convert_to_dict(file_path=json_file, input_type="json"))
    except Exception as e:
        print(f"Error with JSON file: {e}")

    # Test YAML file
    try:
        print("YAML File:", read_and_convert_to_dict(file_path=yaml_file, input_type="yaml"))
    except Exception as e:
        print(f"Error with YAML file: {e}")

    # Test dictionary input
    try:
        print("Dict Input:", read_and_convert_to_dict(input_data=dict_input, input_type="dict"))
    except Exception as e:
        print(f"Error with dict input: {e}")

    # Test JSON string input
    try:
        print("JSON String Input:", read_and_convert_to_dict(input_data=json_input, input_type="json"))
    except Exception as e:
        print(f"Error with JSON string input: {e}")

    # Test YAML string input
    try:
        print("YAML String Input:", read_and_convert_to_dict(input_data=yaml_input, input_type="yaml"))
    except Exception as e:
        print(f"Error with YAML string input: {e}")
