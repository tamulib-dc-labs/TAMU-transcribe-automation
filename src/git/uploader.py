"""
Git uploader for pushing transcription results to GitHub repository.
"""

import os
import subprocess
from datetime import datetime
from typing import Optional


class GitUploader:
    """Handles uploading files to GitHub repository via git operations."""
    
    def __init__(
        self,
        source_folder: str,
        repo_folder: str,
        owner: str,
        repo_name: str,
        username: str,
        token: str,
        branch_prefix: str = "upload"
    ):
        """
        Initialize Git uploader.
        
        Args:
            source_folder: Path to folder containing files to upload
            repo_folder: Path where git repo will be cloned/maintained
            owner: GitHub repository owner
            repo_name: GitHub repository name
            username: GitHub username for authentication
            token: GitHub personal access token
            branch_prefix: Prefix for branch name (default: "upload")
        """
        self.source_folder = source_folder
        self.repo_folder = repo_folder
        self.owner = owner
        self.repo_name = repo_name
        self.username = username
        self.token = token
        self.branch_prefix = branch_prefix
        self.remote_url = f"https://{username}:{token}@github.com/{owner}/{repo_name}.git"
    
    def _run_git_command(self, args: list, cwd: Optional[str] = None) -> bool:
        """
        Run a git command.
        
        Args:
            args: Command arguments
            cwd: Working directory (default: repo_folder)
            
        Returns:
            bool: True if successful
        """
        try:
            result = subprocess.run(
                args,
                cwd=cwd or self.repo_folder,
                check=True,
                text=True,
                capture_output=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Git command failed: {' '.join(args)}")
            print(f"Return code: {e.returncode}")
            if e.stdout:
                print(f"stdout: {e.stdout}")
            if e.stderr:
                print(f"stderr: {e.stderr}")
            return False
    
    def setup_repository(self) -> bool:
        """
        Clone or update the git repository.
        
        Returns:
            bool: True if successful
        """
        if not os.path.exists(self.repo_folder):
            print("Cloning repository (first time setup)...")
            
            # Create parent directory if needed
            os.makedirs(os.path.dirname(self.repo_folder), exist_ok=True)
            
            # Clone repository
            try:
                subprocess.run(
                    ["git", "clone", self.remote_url, self.repo_folder],
                    check=True,
                    capture_output=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Failed to clone repository: {e.stderr}")
                return False
            
            # Disable credential helper
            subprocess.run(
                ["git", "config", "--local", "--unset", "credential.helper"],
                cwd=self.repo_folder,
                check=False
            )
        else:
            print("Updating existing repository...")
            
            # Update remote URL with new token
            print("Refreshing remote credentials...")
            if not self._run_git_command(["git", "remote", "set-url", "origin", self.remote_url]):
                return False
            
            # Checkout main and pull latest.
            #
            # A previous run rsyncs json/ and vtts/ into this folder before
            # committing. If that run did not finish, those files are left
            # untracked and git refuses to check out main over them. They are
            # regenerated from the output folder by sync_files() moments later,
            # so taking main's copy is safe.
            if not self._run_git_command(["git", "checkout", "main"]):
                print(
                    "Checkout blocked by leftover files from a previous run; "
                    "taking the committed versions (sync_files re-adds ours next)."
                )
                if not self._run_git_command(["git", "checkout", "-f", "main"]):
                    return False

            if not self._run_git_command(["git", "pull", "origin", "main"]):
                return False
        
        return True
    
    def create_branch(self) -> str:
        """
        Create a new timestamped branch.
        
        Returns:
            str: Branch name
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        branch_name = f"{self.branch_prefix}-{timestamp}"
        
        print(f"Creating branch: {branch_name}")
        self._run_git_command(["git", "checkout", "-b", branch_name])
        
        return branch_name
    
    def sync_files(self) -> bool:
        """
        Sync files from source to git repository using rsync.
        
        Returns:
            bool: True if successful
        """
        print("SYNCING FILES...")
        print("Copying from Source -> Git Repo (additions/updates only)...")
        
        # Use rsync to copy files without deleting existing files
        cmd_rsync = [
            "rsync", "-av",
            "--exclude", ".git",
            f"{self.source_folder}/",
            f"{self.repo_folder}/"
        ]
        
        try:
            subprocess.run(cmd_rsync, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to sync files: {e}")
            return False
    
    def commit_and_push(self, branch_name: str) -> bool:
        """
        Commit changes and push to remote.
        
        Args:
            branch_name: Name of branch to push
            
        Returns:
            bool: True if successful
        """
        # Stage changes (only additions and modifications, no deletions)
        print("Staging changes (new and modified files only)...")
        if not self._run_git_command(["git", "add", "."]):
            return False
        
        # Check if there are changes to commit
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.repo_folder,
            capture_output=True,
            text=True
        )
        
        if not result.stdout.strip():
            print("No new changes found after copying. Nothing to push.")
            return True
        
        # Commit
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        print("Committing...")
        if not self._run_git_command(["git", "commit", "-m", f"Upload via automation {timestamp}"]):
            return False
        
        # Push
        print(f"Pushing to {branch_name}...")
        if not self._run_git_command(["git", "push", "-u", "origin", branch_name]):
            return False
        
        print(f"SUCCESS! Pull Request URL: https://github.com/{self.owner}/{self.repo_name}/pull/new/{branch_name}")
        return True
    
    def upload(self) -> bool:
        """
        Execute the complete upload pipeline.
        
        Returns:
            bool: True if successful
        """
        print(f"Starting Git Upload Pipeline")
        print(f"Source Data: {self.source_folder}")
        print(f"Git Repo Dir: {self.repo_folder}")
        
        # Step 1: Setup repository
        if not self.setup_repository():
            return False
        
        # Step 2: Create new branch
        branch_name = self.create_branch()
        
        # Step 3: Sync files
        if not self.sync_files():
            return False
        
        # Step 4: Commit and push
        if not self.commit_and_push(branch_name):
            return False
        
        return True
    
    def get_uploaded_file_urls(self, folder_names: list = None) -> dict:
        """
        Build GitHub Pages URLs for uploaded files.
        
        Args:
            folder_names: List of folder names (audio file base names) to get URLs for
        
        Returns:
            dict: Mapping of name -> {json_url, vtt_url}
        """
        import os
        
        # GitHub Pages URL format
        base_url = f"https://{self.owner}.github.io/{self.repo_name}"
        
        file_urls = {}
        
        # JSON files are in source_folder/json/
        # VTT files are in source_folder/vtts/
        json_dir = os.path.join(self.source_folder, "json")
        vtt_dir = os.path.join(self.source_folder, "vtts")
        
        # Get available JSON files
        json_files = {}
        if os.path.exists(json_dir):
            for f in os.listdir(json_dir):
                if f.endswith('.json'):
                    # Extract base name (without extension)
                    base_name = os.path.splitext(f)[0]
                    json_files[base_name] = f
        
        # Get available VTT files
        vtt_files = {}
        if os.path.exists(vtt_dir):
            for f in os.listdir(vtt_dir):
                if f.endswith('.vtt'):
                    base_name = os.path.splitext(f)[0]
                    vtt_files[base_name] = f
        
        # If folder_names provided, use those; otherwise use all found files
        if folder_names is None:
            folder_names = list(set(list(json_files.keys()) + list(vtt_files.keys())))
        
        for name in folder_names:
            json_url = None
            vtt_url = None
            
            # Look for matching JSON file
            if name in json_files:
                json_url = f"{base_url}/json/{json_files[name]}"
            
            # Look for matching VTT file
            if name in vtt_files:
                vtt_url = f"{base_url}/vtts/{vtt_files[name]}"
            
            if json_url or vtt_url:
                file_urls[name] = {
                    "json_url": json_url,
                    "vtt_url": vtt_url
                }
        
        return file_urls

