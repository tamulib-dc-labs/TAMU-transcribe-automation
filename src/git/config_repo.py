"""
Config repository manager for handling edge-grant-reviewer repo operations.

This module handles:
- Cloning/updating the config repository
- Reading config-to-process.json
- Updating config.json with new entries
"""

import os
import json
import subprocess
from datetime import datetime
from typing import Optional, List, Dict, Any


class ConfigRepoManager:
    """Handles operations on the edge-grant-reviewer config repository."""
    
    def __init__(
        self,
        repo_folder: str,
        owner: str,
        repo_name: str,
        username: str,
        token: str,
        config_json_path: str = "public/config-to-process.json",
        output_config_path: str = "public/config.json"
    ):
        """
        Initialize Config Repo Manager.
        
        Args:
            repo_folder: Path where git repo will be cloned/maintained
            owner: GitHub repository owner
            repo_name: GitHub repository name
            username: GitHub username for authentication
            token: GitHub personal access token
            config_json_path: Path to config-to-process.json within repo
            output_config_path: Path to config.json to update within repo
        """
        self.repo_folder = repo_folder
        self.owner = owner
        self.repo_name = repo_name
        self.username = username
        self.token = token
        self.config_json_path = config_json_path
        self.output_config_path = output_config_path
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
        Clone or update the config git repository.
        
        Returns:
            bool: True if successful
        """
        if not os.path.exists(self.repo_folder):
            print(f"Cloning config repository {self.repo_name}...")
            
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
            print(f"Updating existing config repository {self.repo_name}...")
            
            # Update remote URL with new token
            print("Refreshing remote credentials...")
            if not self._run_git_command(["git", "remote", "set-url", "origin", self.remote_url]):
                return False
            
            # Checkout main and pull latest
            if not self._run_git_command(["git", "checkout", "main"]):
                return False
            
            if not self._run_git_command(["git", "pull", "origin", "main"]):
                return False
        
        return True
    
    def get_processed_names(self) -> set:
        """
        Get set of names that have already been processed (exist in config.json).
        
        Returns:
            Set of name strings that are already in config.json
        """
        config_path = os.path.join(self.repo_folder, self.output_config_path)
        
        if not os.path.exists(config_path):
            return set()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                existing_entries = json.load(f)
            
            # Extract all names from existing entries
            processed_names = {entry.get("name", "") for entry in existing_entries if entry.get("name")}
            return processed_names
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Could not read config.json for processed names: {e}")
            return set()
    
    def read_config_to_process(self) -> List[Dict[str, Any]]:
        """
        Read and parse config-to-process.json, filtering out already processed entries.
        
        Returns:
            List of config entries that have NOT yet been processed
        """
        config_path = os.path.join(self.repo_folder, self.config_json_path)

        if self.config_json_path == self.output_config_path:
            # The output file is what marks entries as already done. If it is
            # also the input, every entry in it is treated as finished, so the
            # run finds nothing to do - and reports it as "all done" rather
            # than as a misconfiguration.
            print(
                "WARNING: config_json_path and output_config_path are both "
                f"{self.config_json_path!r}.\n"
                "         The output file marks work as done, so every entry in "
                "it is skipped --\n"
                "         which means this run will find nothing to transcribe. "
                "Use two files:\n"
                "         public/config-to-process.json (input) and "
                "public/config.json (output)."
            )

        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            return []
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            total_entries = len(config_data)
            print(f"Read {total_entries} entries from {self.config_json_path}")
            
            # Filter out already processed entries
            processed_names = self.get_processed_names()
            if processed_names:
                print(f"Found {len(processed_names)} already-processed entries in "
                      f"{self.output_config_path}")
                
                new_entries = []
                skipped = 0
                for entry in config_data:
                    name = entry.get("name", "")
                    if name in processed_names:
                        skipped += 1
                    else:
                        new_entries.append(entry)
                
                if skipped > 0:
                    print(f"Skipping {skipped} already processed entries")
                print(f"Remaining entries to process: {len(new_entries)}")
                return new_entries
            
            return config_data
        except json.JSONDecodeError as e:
            print(f"Error parsing config JSON: {e}")
            return []
        except Exception as e:
            print(f"Error reading config file: {e}")
            return []
    
    def update_config_json(self, new_entries: List[Dict[str, Any]]) -> bool:
        """
        Append new entries to config.json.
        
        Args:
            new_entries: List of entry dicts with audio, url, vtt, name fields
            
        Returns:
            bool: True if successful
        """
        config_path = os.path.join(self.repo_folder, self.output_config_path)
        
        # Read existing config
        existing_entries = []
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    existing_entries = json.load(f)
                print(f"Read {len(existing_entries)} existing entries from config.json")
            except (json.JSONDecodeError, Exception) as e:
                print(f"Warning: Could not read existing config.json: {e}")
                existing_entries = []
        
        # Append new entries
        updated_entries = existing_entries + new_entries
        
        # Write updated config
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(updated_entries, f, indent=2)
            print(f"Updated config.json with {len(new_entries)} new entries (total: {len(updated_entries)})")
            return True
        except Exception as e:
            print(f"Error writing config.json: {e}")
            return False
    
    def commit_and_push(self, message: Optional[str] = None) -> bool:
        """
        Commit changes and push to a new branch.
        
        Args:
            message: Commit message (default: auto-generated with timestamp)
            
        Returns:
            bool: True if successful
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        branch_name = f"transcription-update-{timestamp}"
        
        # Create and checkout new branch
        print(f"Creating new branch: {branch_name}")
        if not self._run_git_command(["git", "checkout", "-b", branch_name]):
            return False
        
        # Stage changes
        print("Staging config.json changes...")
        if not self._run_git_command(["git", "add", self.output_config_path]):
            return False
        
        # Check if there are changes to commit
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.repo_folder,
            capture_output=True,
            text=True
        )
        
        if not result.stdout.strip():
            print("No changes to config.json. Nothing to push.")
            # Switch back to main
            self._run_git_command(["git", "checkout", "main"])
            return True
        
        # Commit
        if message is None:
            message = f"Update config.json via automation {timestamp}"
        
        print(f"Committing: {message}")
        if not self._run_git_command(["git", "commit", "-m", message]):
            return False
        
        # Push to new branch
        print(f"Pushing to branch {branch_name}...")
        if not self._run_git_command(["git", "push", "-u", "origin", branch_name]):
            return False
        
        print(f"SUCCESS! Config updated in branch {branch_name}")
        print(f"Create PR at: https://github.com/{self.owner}/{self.repo_name}/pull/new/{branch_name}")
        return True
