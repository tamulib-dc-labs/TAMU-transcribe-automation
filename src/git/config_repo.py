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
        """Names already listed in output_config_path.

        Used when writing that file, to replace an entry rather than add a
        second copy of it. It is NOT used to decide what to transcribe - see
        read_config_to_process().
        """
        config_path = os.path.join(self.repo_folder, self.output_config_path)

        if not os.path.exists(config_path):
            return set()

        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                existing_entries = json.load(handle)
            return {e.get("name") for e in existing_entries if e.get("name")}
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: could not read {self.output_config_path}: {exc}")
            return set()

    def read_config_to_process(self) -> List[Dict[str, Any]]:
        """Read the work list from config_json_path.

        Returns every entry. It deliberately does NOT filter against
        output_config_path: that file is written *after* a run so the reviewer
        app has links to the transcripts, and it is not a record of what has
        been transcribed.

        What has already been done is decided by the transcripts repository
        instead - see TranscriptionPipeline._collect_completed(). That record
        survives a /scratch purge and does not depend on the previous run
        having finished its final step.
        """
        config_path = os.path.join(self.repo_folder, self.config_json_path)

        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            return []

        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                config_data = json.load(handle)
        except json.JSONDecodeError as exc:
            print(f"Could not parse {self.config_json_path}: {exc}")
            return []

        print(f"Read {len(config_data)} entries from {self.config_json_path}")

        if self.config_json_path == self.output_config_path:
            print(
                f"Note: {self.config_json_path} is both the work list and the "
                "file written afterwards,"
            )
            print("      so this run will replace it with the transcript links.")

        return config_data

    def update_config_json(self, new_entries: List[Dict[str, Any]]) -> bool:
        """Write the transcript links into output_config_path.

        Entries are merged by name, not appended: re-transcribing an interview
        replaces its row instead of adding a second one. Order is preserved,
        so the reviewer app's list does not reshuffle between runs.
        """
        config_path = os.path.join(self.repo_folder, self.output_config_path)

        existing_entries: List[Dict[str, Any]] = []
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as handle:
                    existing_entries = json.load(handle)
                print(
                    f"Read {len(existing_entries)} existing entries from "
                    f"{self.output_config_path}"
                )
            except (json.JSONDecodeError, OSError) as exc:
                print(f"Warning: could not read existing config: {exc}")
                existing_entries = []

        replacements = {e["name"]: e for e in new_entries if e.get("name")}

        merged = []
        replaced = set()
        for entry in existing_entries:
            name = entry.get("name")
            if name in replacements:
                merged.append(replacements[name])
                replaced.add(name)
            else:
                merged.append(entry)

        added = [e for e in new_entries if e.get("name") not in replaced]
        merged.extend(added)

        try:
            os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(merged, handle, indent=2)
        except OSError as exc:
            print(f"Error writing {self.output_config_path}: {exc}")
            return False

        updated = len(new_entries) - len(added)
        print(
            f"Wrote {self.output_config_path}: {len(added)} new, "
            f"{updated} updated, {len(merged)} total"
        )
        return True

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
