# script.py
from git import Repo
from pathlib import Path

def run_script(repo_url: str, branch: str, file_content: str, clone_dir: Path) -> None:
    # Clone the repository
    repo = Repo.clone_from(repo_url, clone_dir)
    
    # Modify a file
    file_path = clone_dir / "example.txt"
    with open(file_path, "w") as f:
        f.write(file_content)
    
    # Stage, commit, and push changes
    repo.git.add("--all")
    repo.git.commit("-m", "Update example.txt")
    repo.git.push("origin", branch)



# tests/test_script.py
from pathlib import Path
import pytest
from script import run_script
from unittest.mock import MagicMock, patch

def test_run_script(mocker, tmp_path):
    # Mock Repo.clone_from and its return value
    mock_repo = MagicMock()
    mock_clone_from = mocker.patch(
        "git.Repo.clone_from",
        return_value=mock_repo
    )

    # Test inputs
    repo_url = "https://github.com/example/repo.git"
    branch = "main"
    file_content = "test content"
    clone_dir = tmp_path / "clone"

    # Run the function
    run_script(repo_url, branch, file_content, clone_dir)

    # Assertions
    # 1. Verify clone_from was called with the correct args
    mock_clone_from.assert_called_once_with(repo_url, str(clone_dir))

    # 2. Verify file was written to the correct path
    file_path = clone_dir / "example.txt"
    assert file_path.exists()
    assert file_path.read_text() == file_content

    # 3. Verify Git operations were called correctly
    mock_repo.git.add.assert_called_once_with("--all")
    mock_repo.git.commit.assert_called_once_with("-m", "Update example.txt")
    mock_repo.git.push.assert_called_once_with("origin", branch)


def test_clone_failure(mocker, tmp_path):
    # Simulate a clone failure
    mocker.patch(
        "git.Repo.clone_from",
        side_effect=Exception("Clone failed")
    )

    with pytest.raises(Exception, match="Clone failed"):
        run_script(
            repo_url="invalid-url",
            branch="main",
            file_content="test",
            clone_dir=tmp_path / "clone"
        )
