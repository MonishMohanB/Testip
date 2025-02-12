import pytest
import json
import yaml
from your_module import read_and_convert_to_dict  # Replace 'your_module' with the actual module name

# Helper function to create temporary files
def create_temp_file(tmp_path, content, file_type):
    file_path = tmp_path / f"temp.{file_type}"
    with open(file_path, "w") as file:
        file.write(content)
    return file_path

# Test cases
def test_json_file(tmp_path):
    """Test reading and converting a JSON file."""
    json_content = '{"name": "John", "age": 30}'
    file_path = create_temp_file(tmp_path, json_content, "json")
    result = read_and_convert_to_dict(file_path=file_path, input_type="json")
    assert result == {"name": "John", "age": 30}

def test_yaml_file(tmp_path):
    """Test reading and converting a YAML file."""
    yaml_content = """
    name: John
    age: 30
    """
    file_path = create_temp_file(tmp_path, yaml_content, "yaml")
    result = read_and_convert_to_dict(file_path=file_path, input_type="yaml")
    assert result == {"name": "John", "age": 30}

def test_dict_input():
    """Test passing a dictionary directly."""
    dict_input = {"name": "John", "age": 30}
    result = read_and_convert_to_dict(input_data=dict_input, input_type="dict")
    assert result == {"name": "John", "age": 30}

def test_json_string_input():
    """Test passing a JSON string directly."""
    json_input = '{"name": "John", "age": 30}'
    result = read_and_convert_to_dict(input_data=json_input, input_type="json")
    assert result == {"name": "John", "age": 30}

def test_yaml_string_input():
    """Test passing a YAML string directly."""
    yaml_input = """
    name: John
    age: 30
    """
    result = read_and_convert_to_dict(input_data=yaml_input, input_type="yaml")
    assert result == {"name": "John", "age": 30}

def test_invalid_json_file(tmp_path):
    """Test invalid JSON file."""
    invalid_json_content = '{"name": "John", "age": 30'  # Missing closing brace
    file_path = create_temp_file(tmp_path, invalid_json_content, "json")
    with pytest.raises(ValueError, match="Invalid JSON format"):
        read_and_convert_to_dict(file_path=file_path, input_type="json")

def test_invalid_yaml_file(tmp_path):
    """Test invalid YAML file."""
    invalid_yaml_content = """
    name: John
    age: : 30  # Invalid YAML syntax
    """
    file_path = create_temp_file(tmp_path, invalid_yaml_content, "yaml")
    with pytest.raises(ValueError, match="Invalid YAML format"):
        read_and_convert_to_dict(file_path=file_path, input_type="yaml")

def test_file_not_found():
    """Test non-existent file."""
    with pytest.raises(FileNotFoundError):
        read_and_convert_to_dict(file_path="non_existent_file.json", input_type="json")

def test_invalid_input_type():
    """Test invalid input type."""
    with pytest.raises(ValueError, match="Invalid input_type"):
        read_and_convert_to_dict(input_data='{"name": "John", "age": 30}', input_type="invalid_type")

def test_missing_input():
    """Test missing input arguments."""
    with pytest.raises(ValueError, match="Either file_path or input_data must be provided"):
        read_and_convert_to_dict(input_type="json")

def test_conflicting_input():
    """Test conflicting input arguments."""
    with pytest.raises(ValueError, match="Only one of file_path or input_data should be provided"):
        read_and_convert_to_dict(file_path="data.json", input_data='{"name": "John"}', input_type="json")

def test_invalid_dict_input():
    """Test invalid dictionary input."""
    with pytest.raises(ValueError, match="input_data must be a dictionary for input_type 'dict'"):
        read_and_convert_to_dict(input_data='{"name": "John"}', input_type="dict")

def test_invalid_json_string_input():
    """Test invalid JSON string input."""
    with pytest.raises(ValueError, match="Invalid JSON format"):
        read_and_convert_to_dict(input_data='{"name": "John", "age": 30', input_type="json")

def test_invalid_yaml_string_input():
    """Test invalid YAML string input."""
    with pytest.raises(ValueError, match="Invalid YAML format"):
        read_and_convert_to_dict(input_data="name: John\nage: : 30", input_type="yaml")
