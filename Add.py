import git
import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime

class GitRepoProcessor:
    """Git repository operations handler with datetime commit messages"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.repo = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate_config()
        
    def _validate_config(self):
        """Validate essential configuration parameters"""
        required_keys = ['target_path', 'target_branch', 'subdir', 'files_to_add']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def prepare_repository(self):
        """Prepare repository source with error handling"""
        try:
            target = Path(self.config['target_path'])
            if target.exists():
                self.logger.warning(f"Cleaning existing target: {target}")
                shutil.rmtree(target)
            
            if self.config.get('use_existing_repo'):
                self._copy_existing()
            else:
                self._clone_repo()
            
            self.repo = git.Repo(self.config['target_path'])
            return True
        except Exception as e:
            self.logger.error(f"Repository preparation failed: {e}", exc_info=True)
            return False

    def _copy_existing(self):
        """Copy local repository with validation"""
        src = Path(self.config.get('existing_repo_path'))
        if not src.exists():
            raise FileNotFoundError(f"Source repository missing: {src}")
        shutil.copytree(src, self.config['target_path'])

    def _clone_repo(self):
        """Clone remote repository via SSH"""
        if 'repo_url' not in self.config:
            raise ValueError("SSH URL required for cloning")
        git.Repo.clone_from(
            self.config['repo_url'],
            self.config['target_path'],
            branch=self.config.get('clone_branch', 'master')
        )

    def execute_workflow(self):
        """Full workflow executor with datetime commit messages"""
        steps = [
            self.checkout_branch,
            self.create_files,
            self.commit_changes,
            self.push_changes
        ]
        
        for step in steps:
            if not step():
                return False
        return True

    def checkout_branch(self):
        """Checkout/create target branch"""
        try:
            self.repo.git.checkout('-B', self.config['target_branch'])
            self.logger.info(f"Checked out branch: {self.config['target_branch']}")
            return True
        except git.exc.GitCommandError as e:
            self.logger.error(f"Branch operation failed: {e}")
            return False

    def create_files(self):
        """Create files in configured subdirectory"""
        try:
            subdir = Path(self.repo.working_dir) / self.config['subdir']
            subdir.mkdir(parents=True, exist_ok=True)
            
            for filename in self.config['files_to_add']:
                filepath = subdir / filename
                filepath.write_text(f"# Auto-generated at {self._current_time()}\n")
                self.logger.debug(f"Created file: {filepath}")
            
            return True
        except Exception as e:
            self.logger.error(f"File creation failed: {e}", exc_info=True)
            return False

    def commit_changes(self, custom_message: str = None):
        """Commit changes with timestamp"""
        try:
            self.repo.git.add(A=True)
            commit_msg = self._format_commit_message(custom_message)
            self.repo.index.commit(commit_msg)
            self.logger.info(f"Committed: {commit_msg}")
            return True
        except git.exc.GitCommandError as e:
            self.logger.error(f"Commit failed: {e}")
            return False

    def _format_commit_message(self, custom_msg: str = None) -> str:
        """Generate standardized commit message with timestamp"""
        timestamp = self._current_time()
        base_msg = custom_msg or f"Add files to {self.config['subdir']}"
        return f"{base_msg} - {timestamp}"

    def _current_time(self) -> str:
        """Get current datetime in standardized format"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def push_changes(self):
        """Push changes to remote repository"""
        try:
            origin = self.repo.remote('origin')
            push_result = origin.push(refspec=f"{self.config['target_branch']}:{self.config['target_branch']}")
            
            for result in push_result:
                if result.flags & result.ERROR:
                    raise RuntimeError(f"Push error: {result.summary}")
            
            self.logger.info("Successfully pushed changes")
            return True
        except Exception as e:
            self.logger.error(f"Push failed: {e}")
            return False

# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = {
        "use_existing_repo": False,
        "repo_url": "git@github.com:user/repo.git",
        "target_path": "./repo-clone",
        "target_branch": "auto-commit-branch",
        "subdir": "generated_files",
        "files_to_add": ["data.txt", "config.json"]
    }
    
    processor = GitRepoProcessor(config)
    
    if processor.prepare_repository() and processor.execute_workflow():
        logging.info("Workflow completed successfully")
    else:
        logging.error("Workflow failed")
        exit(1)


import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from datetime import datetime
from git_repo_processor import GitRepoProcessor  # Update with your module name

@pytest.fixture
def base_config():
    return {
        "target_path": "/tmp/test-repo",
        "target_branch": "test-branch",
        "subdir": "test-dir",
        "files_to_add": ["test-file.txt"],
        "repo_url": "git@github.com:user/repo.git"
    }

@pytest.fixture
def mock_repo():
    repo = Mock()
    repo.working_dir = "/tmp/test-repo"
    repo.git = Mock()
    repo.index = Mock()
    repo.remote.return_value = Mock()
    return repo

def test_initialization_valid_config(base_config):
    processor = GitRepoProcessor(base_config)
    assert processor.config == base_config

def test_initialization_missing_config_key(base_config):
    del base_config["target_path"]
    with pytest.raises(ValueError):
        GitRepoProcessor(base_config)

@patch("shutil.rmtree")
@patch("git.Repo.clone_from")
def test_prepare_repository_clone(mock_clone, mock_rmtree, base_config):
    base_config["use_existing_repo"] = False
    processor = GitRepoProcessor(base_config)
    
    result = processor.prepare_repository()
    
    mock_rmtree.assert_called_once_with(Path("/tmp/test-repo"))
    mock_clone.assert_called_once_with(
        "git@github.com:user/repo.git",
        "/tmp/test-repo",
        branch="master"
    )
    assert result is True

@patch("shutil.rmtree")
@patch("shutil.copytree")
def test_prepare_repository_copy(mock_copytree, mock_rmtree, base_config):
    base_config["use_existing_repo"] = True
    base_config["existing_repo_path"] = "/tmp/existing-repo"
    processor = GitRepoProcessor(base_config)
    
    result = processor.prepare_repository()
    
    mock_rmtree.assert_called_once_with(Path("/tmp/test-repo"))
    mock_copytree.assert_called_once_with("/tmp/existing-repo", "/tmp/test-repo")
    assert result is True

@patch("builtins.open")
@patch("pathlib.Path.mkdir")
def test_create_files(mock_mkdir, mock_open, base_config, mock_repo):
    processor = GitRepoProcessor(base_config)
    processor.repo = mock_repo
    
    result = processor.create_files()
    
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_open.assert_called()
    assert result is True

@patch("datetime.datetime")
def test_commit_message_format(mock_datetime, base_config, mock_repo):
    fixed_time = datetime(2023, 1, 1, 12, 0, 0)
    mock_datetime.now.return_value = fixed_time
    
    processor = GitRepoProcessor(base_config)
    processor.repo = mock_repo
    
    processor.commit_changes()
    expected_msg = f"Add files to test-dir - {fixed_time.strftime('%Y-%m-%d %H:%M:%S')}"
    mock_repo.index.commit.assert_called_once_with(expected_msg)

def test_checkout_branch_success(base_config, mock_repo):
    processor = GitRepoProcessor(base_config)
    processor.repo = mock_repo
    
    result = processor.checkout_branch()
    
    mock_repo.git.checkout.assert_called_once_with("-B", "test-branch")
    assert result is True

def test_checkout_branch_failure(base_config, mock_repo):
    processor = GitRepoProcessor(base_config)
    processor.repo = mock_repo
    mock_repo.git.checkout.side_effect = Exception("Git error")
    
    result = processor.checkout_branch()
    
    assert result is False

@patch("git.Repo")
def test_full_workflow(mock_repo_class, base_config, tmp_path):
    mock_repo = Mock()
    mock_repo_class.clone_from.return_value = mock_repo
    mock_repo.remote.return_value.push.return_value = [Mock(flags=0)]
    
    base_config["target_path"] = str(tmp_path / "repo")
    processor = GitRepoProcessor(base_config)
    
    assert processor.prepare_repository() is True
    assert processor.checkout_branch() is True
    assert processor.create_files() is True
    assert processor.commit_changes() is True
    assert processor.push_changes() is True

@patch("logging.Logger.error")
def test_error_handling(mock_log_error, base_config):
    processor = GitRepoProcessor(base_config)
    with patch("shutil.rmtree", side_effect=Exception("Test error")):
        result = processor.prepare_repository()
        assert result is False
        mock_log_error.assert_called()




