import os
import git

def manage_repo(dir_path, target_path, push_repo_name, branch_name):
    # Hardcoded repo URL (example)
    repo_url = "https://github.com/your_user/your_repo.git"
    
    # Clone the repo if target_path is None
    if target_path is None:
        # Create the directory for push_repo_name under dir_path
        target_path = os.path.join(dir_path, push_repo_name)
        
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        
        # Clone the repository
        repo = git.Repo.clone_from(repo_url, target_path, branch=branch_name)
    else:
        # If target_path is provided, open the existing repo
        repo = git.Repo(target_path)
    
    # Navigate to the folder inside the repository (assuming it's named 'folder_name')
    folder_name = "folder_name"
    folder_path = os.path.join(target_path, folder_name)
    
    # Alter the contents of the folder (for example, creating a new file or modifying one)
    with open(os.path.join(folder_path, "new_file.txt"), "w") as f:
        f.write("This is a new file created during the test.")
    
    # Add the changes, commit, and push to the repository
    repo.git.add(A=True)
    repo.git.commit(m="Altered folder contents and added new file.")
    origin = repo.remotes.origin
    origin.push()

    return f"Changes pushed to {repo_url} on branch {branch_name}"

import pytest
from unittest.mock import patch, MagicMock
import os

# Import the function to be tested
from your_module import manage_repo

@pytest.fixture
def mock_git_repo():
    """Fixture to mock the git.Repo object"""
    mock_repo = MagicMock()
    mock_repo.git.add = MagicMock()
    mock_repo.git.commit = MagicMock()
    mock_repo.remotes.origin.push = MagicMock()
    return mock_repo

@patch("git.Repo.clone_from")
@patch("git.Repo")
def test_manage_repo(mock_git_repo_class, mock_clone_from, mock_git_repo):
    # Setup mock repo clone
    mock_clone_from.return_value = mock_git_repo
    
    # Example inputs
    dir_path = "/tmp/test_dir"
    target_path = None
    push_repo_name = "my_repo"
    branch_name = "main"
    
    # Call the function
    result = manage_repo(dir_path, target_path, push_repo_name, branch_name)
    
    # Validate the clone method is called once
    mock_clone_from.assert_called_once_with(
        "https://github.com/your_user/your_repo.git", 
        os.path.join(dir_path, push_repo_name), 
        branch=branch_name
    )
    
    # Validate the repository methods are invoked
    mock_git_repo.git.add.assert_called_once_with(A=True)
    mock_git_repo.git.commit.assert_called_once_with(m="Altered folder contents and added new file.")
    mock_git_repo.remotes.origin.push.assert_called_once()

    # Check the final result
    assert result == f"Changes pushed to https://github.com/your_user/your_repo.git on branch {branch_name}"


