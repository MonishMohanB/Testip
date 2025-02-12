import pytest
import json
import yaml
from tempfile import NamedTemporaryFile

# Function to test
def load_to_dict(path: str, input_type: str) -> dict:
    """
    Load data from a file (JSON, YAML) or directly use a dictionary and return a Python dictionary.

    :param path: Path to the file or the dictionary itself.
    :param input_type: Type of input ('json', 'yaml', or 'dict').
    :return: Python dictionary.
    """
    if input_type == 'dict':
        return path if isinstance(path, dict) else {}

    with open(path, 'r') as file:
        if input_type == 'json':
            return json.load(file)
        elif input_type == 'yaml':
            return yaml.safe_load(file)
        else:
            raise ValueError("Unsupported input type. Use 'json', 'yaml', or 'dict'.")

# Test cases
def test_load_json():
    # Create a temporary JSON file
    with NamedTemporaryFile(mode='w', delete=False) as temp_file:
        json.dump({"key": "value"}, temp_file)
        temp_file_path = temp_file.name

    # Test loading JSON
    result = load_to_dict(temp_file_path, 'json')
    assert result == {"key": "value"}

def test_load_yaml():
    # Create a temporary YAML file
    with NamedTemporaryFile(mode='w', delete=False) as temp_file:
        yaml.dump({"key": "value"}, temp_file)
        temp_file_path = temp_file.name

    # Test loading YAML
    result = load_to_dict(temp_file_path, 'yaml')
    assert result == {"key": "value"}

def test_load_dict():
    # Test loading dictionary directly
    input_dict = {"key": "value"}
    result = load_to_dict(input_dict, 'dict')
    assert result == {"key": "value"}

def test_unsupported_input_type():
    # Test unsupported input type
    with pytest.raises(ValueError, match="Unsupported input type. Use 'json', 'yaml', or 'dict'."):
        load_to_dict("dummy_path", 'unsupported_type')

def test_invalid_json_file():
    # Create an invalid JSON file
    with NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write("invalid json")
        temp_file_path = temp_file.name

    # Test loading invalid JSON
    with pytest.raises(json.JSONDecodeError):
        load_to_dict(temp_file_path, 'json')

def test_invalid_yaml_file():
    # Create an invalid YAML file
    with NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write("invalid: yaml: :")
        temp_file_path = temp_file.name

    # Test loading invalid YAML
    with pytest.raises(yaml.YAMLError):
        load_to_dict(temp_file_path, 'yaml')
