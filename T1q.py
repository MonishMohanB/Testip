import os
import shutil
import pytest
from copy_directory_contents import copy_directory_contents

@pytest.fixture
def setup_directories():
    """
    Creates temporary source and target directories for testing.
    """
    src_dir = 'test_source'
    target_dir = 'test_target'

    # Create a source directory and add some files
    os.makedirs(src_dir, exist_ok=True)
    with open(os.path.join(src_dir, 'file1.txt'), 'w') as f:
        f.write('Test file 1')

    with open(os.path.join(src_dir, 'file2.txt'), 'w') as f:
        f.write('Test file 2')

    yield src_dir, target_dir

    # Clean up
    if os.path.exists(src_dir):
        shutil.rmtree(src_dir)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

def test_copy_directory_contents(setup_directories):
    src_dir, target_dir = setup_directories
    
    # Call the function to copy contents
    copy_directory_contents(src_dir, target_dir)
    
    # Test that the target directory now contains the copied files
    assert os.path.exists(target_dir)
    assert os.path.exists(os.path.join(target_dir, 'file1.txt'))
    assert os.path.exists(os.path.join(target_dir, 'file2.txt'))

def test_overwrite_target_directory(setup_directories):
    src_dir, target_dir = setup_directories
    
    # Create a target directory and add a test file
    os.makedirs(target_dir, exist_ok=True)
    with open(os.path.join(target_dir, 'existing_file.txt'), 'w') as f:
        f.write('This is an existing file.')

    # Call the function to copy contents
    copy_directory_contents(src_dir, target_dir)
    
    # Test that the existing file in the target directory is overwritten
    assert not os.path.exists(os.path.join(target_dir, 'existing_file.txt'))
    assert os.path.exists(os.path.join(target_dir, 'file1.txt'))
    assert os.path.exists(os.path.join(target_dir, 'file2.txt'))

def test_invalid_source_directory():
    # Test for invalid source directory
    with pytest.raises(ValueError):
        copy_directory_contents('invalid_source_dir', 'test_target')

def test_empty_source_directory():
    # Test if source directory is empty
    src_dir = 'empty_source'
    target_dir = 'test_target_empty'
    
    os.makedirs(src_dir, exist_ok=True)

    copy_directory_contents(src_dir, target_dir)
    
    # Ensure the target directory is also empty after copy
    assert os.path.exists(target_dir)
    assert not os.listdir(target_dir)

    shutil.rmtree(src_dir)
    shutil.rmtree(target_dir)
  
