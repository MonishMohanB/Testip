import pytest
import yaml
from yaml import YAMLError
from pathlib import Path

# Assuming the function is in a module named 'yaml_parser'
from yaml_parser import extract_entities_from_yaml


def test_extract_entities_valid_entity(tmp_path):
    # Arrange: Create a temporary YAML file with valid data
    yaml_data = {
        'users': [
            {'name': 'Alice', 'id': 1},
            {'name': 'Bob', 'id': 2}
        ],
        'products': [
            {'name': 'Product A', 'price': 100}
        ]
    }
    yaml_file = tmp_path / "test_data.yaml"
    yaml_file.write_text(yaml.dump(yaml_data))

    # Act: Call the function with valid entity
    result = extract_entities_from_yaml(yaml_file, 'users')

    # Assert: Verify correct data is returned
    assert result == yaml_data['users']


def test_extract_entities_invalid_entity(tmp_path):
    # Arrange: Create YAML file without the target entity
    yaml_data = {'products': [{'name': 'Product A'}]}
    yaml_file = tmp_path / "test_data.yaml"
    yaml_file.write_text(yaml.dump(yaml_data))

    # Act: Call with non-existent entity
    result = extract_entities_from_yaml(yaml_file, 'users')

    # Assert: Should return empty list
    assert result == []


def test_extract_entities_invalid_yaml(tmp_path):
    # Arrange: Create invalid YAML file
    invalid_yaml = """
    users:
        - name: Alice
        - name: Bob
      invalid_indentation: true
    """
    yaml_file = tmp_path / "invalid.yaml"
    yaml_file.write_text(invalid_yaml)

    # Act & Assert: Verify YAMLError is raised
    with pytest.raises(YAMLError):
        extract_entities_from_yaml(yaml_file, 'users')


def test_extract_entities_file_not_found():
    # Arrange: Use non-existent file path
    non_existent_file = Path('non_existent.yaml')

    # Act & Assert: Verify FileNotFoundError is raised
    with pytest.raises(FileNotFoundError):
        extract_entities_from_yaml(non_existent_file, 'users')




import pytest
from pathlib import Path
import pandas as pd
import pandas.testing as pd_testing

# Assuming the function is in a module named 'csv_loader'
from csv_loader import load_csv_if_exists


def test_load_csv_valid_data(tmp_path):
    # Arrange: Create a valid CSV file
    csv_content = "name,age\nAlice,30\nBob,25"
    file_path = tmp_path / "valid.csv"
    file_path.write_text(csv_content)

    # Act: Load the CSV
    result = load_csv_if_exists(file_path)

    # Assert: Verify DataFrame matches expected data
    expected = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
    pd_testing.assert_frame_equal(result, expected)


def test_load_csv_missing_file():
    # Arrange: Use non-existent file path
    non_existent_path = Path("non_existent.csv")
    
    # Act
    result = load_csv_if_exists(non_existent_path)
    
    # Assert: Return empty DataFrame for missing files
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_load_csv_empty_file(tmp_path):
    # Arrange: Create empty CSV file
    empty_file = tmp_path / "empty.csv"
    empty_file.touch()  # 0-byte file

    # Act
    result = load_csv_if_exists(empty_file)

    # Assert: Handle empty files gracefully
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_load_csv_invalid_data(tmp_path):
    # Arrange: Create CSV with malformed data
    invalid_content = "name,age\nAlice\nBob,25,NY"  # Inconsistent columns
    file_path = tmp_path / "invalid.csv"
    file_path.write_text(invalid_content)

    # Act & Assert: Verify parsing error is raised
    with pytest.raises(pd.errors.ParserError):
        load_csv_if_exists(file_path)


def test_load_csv_permission_error(tmp_path):
    # Arrange: Create file with no read permissions
    restricted_file = tmp_path / "restricted.csv"
    restricted_file.write_text("data")
    restricted_file.chmod(0o000)  # Remove all permissions

    # Act & Assert: Verify error handling
    with pytest.raises(PermissionError):
        load_csv_if_exists(restricted_file)
    
    # Cleanup: Restore permissions (important for Windows compatibility)
    restricted_file.chmod(0o644)
