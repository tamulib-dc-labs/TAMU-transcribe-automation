"""
File management utilities for the transcription pipeline.
"""

import os
import sys
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from datetime import datetime


class FileManager:
    """Handles file and directory operations for the pipeline."""
    
    @staticmethod
    def clear_directory(directory_path: str, keep_structure: bool = True) -> bool:
        """
        Clear all files in a directory.
        
        Args:
            directory_path: Path to directory to clear
            keep_structure: If True, keeps the directory structure
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(directory_path):
                os.makedirs(directory_path, exist_ok=True)
                print(f"Created directory: {directory_path}")
                return True
            
            # Count items (files AND directories) before deletion
            items = os.listdir(directory_path)
            item_count = len(items)
            
            if item_count == 0:
                print(f"Directory {directory_path} is already empty")
                return True
            
            # Remove all files and subdirectories
            for item in items:
                item_path = os.path.join(directory_path, item)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    print(f"Failed to delete {item_path}: {e}")
                    return False
            
            print(f"Cleared {item_count} items from {directory_path}")
            return True
            
        except Exception as e:
            print(f"Error clearing directory {directory_path}: {e}")
            return False
    
    @staticmethod
    def ensure_directory(directory_path: str) -> bool:
        """
        Ensure directory exists, create if it doesn't.
        
        Args:
            directory_path: Path to directory
            
        Returns:
            bool: True if successful
        """
        try:
            os.makedirs(directory_path, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creating directory {directory_path}: {e}")
            return False
    
    @staticmethod
    def count_files(directory_path: str, pattern: str = "*") -> int:
        """
        Count files in directory matching pattern.
        
        Args:
            directory_path: Path to directory
            pattern: Glob pattern (default: all files)
            
        Returns:
            int: Number of matching files
        """
        try:
            return sum(1 for _ in Path(directory_path).rglob(pattern) if _.is_file())
        except Exception:
            return 0
    
    @staticmethod
    def get_file_list(directory_path: str, extensions: Optional[list] = None) -> list:
        """
        Get list of files in directory with optional extension filter.
        
        Args:
            directory_path: Path to directory
            extensions: List of extensions to filter (e.g., ['.mp3', '.wav'])
            
        Returns:
            list: List of file paths
        """
        try:
            path = Path(directory_path)
            if not path.exists():
                return []
            
            if extensions:
                files = []
                for ext in extensions:
                    files.extend(path.rglob(f"*{ext}"))
                return sorted([str(f) for f in files if f.is_file()])
            else:
                return sorted([str(f) for f in path.rglob("*") if f.is_file()])
        except Exception as e:
            print(f"Error listing files in {directory_path}: {e}")
            return []


class CommandRunner:
    """Executes system commands with logging."""
    
    @staticmethod
    def run(cmd, step_num, description: str, shell: bool = False, env: Optional[dict] = None) -> bool:
        """
        Execute a command and log its completion.
        
        Args:
            cmd: Command to execute (string or list)
            step_num: Step number for logging
            description: Description of the command
            shell: Whether to run in shell mode
            env: Optional environment variables
            
        Returns:
            bool: True if successful, False otherwise
        """
        from src.utils.logger import Logger
        
        Logger.log_step(step_num, description, "STARTED")
        
        # Merge current environment with passed env if needed
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        
        try:
            if shell:
                result = subprocess.run(cmd, shell=True, check=True, 
                                       text=True, executable='/bin/bash', env=run_env)
            else:
                result = subprocess.run(cmd, check=True, text=True, env=run_env)
            
            Logger.log_step(step_num, description, "COMPLETED")
            return True
            
        except subprocess.CalledProcessError as e:
            Logger.log_step(step_num, description, "FAILED")
            print(f"Command failed with exit code {e.returncode}")
            return False
        except Exception as e:
            Logger.log_step(step_num, description, "FAILED")
            print(f"Error executing command: {e}")
            return False
    
    @staticmethod
    def submit_slurm_job(job_file: str, dependency: Optional[str] = None) -> Optional[str]:
        """
        Submit a SLURM job and return the job ID.
        
        Args:
            job_file: Path to SLURM job file
            dependency: Slurm dependency, e.g. "afterok:12345". The job stays
                queued until that one finishes successfully, which is how
                prepare -> transcribe -> publish stay ordered without anything
                waiting on a login node.

        Returns:
            str: Job ID if successful, None otherwise
        """
        command = ["sbatch"]
        if dependency:
            command.append(f"--dependency={dependency}")
        command.append(job_file)

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Extract job ID using regex (format: "Submitted batch job 12345")
            output = result.stdout.strip()
            match = re.search(r"Submitted batch job (\d+)", output)
            
            if match:
                job_id = match.group(1)
                return job_id
            else:
                print(f"ERROR: Could not parse Job ID from sbatch output.")
                return None
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to submit SLURM job: {e}")
            print(f"Error output: {e.stderr}")
            sys.stdout.flush()
            return None
        except Exception as e:
            print(f"Error submitting SLURM job: {e}")
            sys.stdout.flush()
            return None
    
    @staticmethod
    def check_slurm_job_status(job_id: str) -> Optional[str]:
        """
        Check SLURM job status.
        
        Args:
            job_id: SLURM job ID
            
        Returns:
            str: Job status (PENDING, RUNNING, COMPLETED, etc.) or None
        """
        try:
            # Use squeue to check if job is in queue
            result = subprocess.run(
                ["squeue", "--job", job_id],
                capture_output=True,
                text=True,
                check=False
            )
            
            # Check if job_id appears in output (job is still in queue)
            if job_id in result.stdout:
                # Extract actual status from output
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Has header + data
                    # Parse the job line to get status (usually in ST column)
                    job_line = lines[1] if len(lines) > 1 else lines[0]
                    tokens = job_line.split()
                    # Try to find status in common positions
                    for token in tokens:
                        if token in ["PD", "R", "CG", "PENDING", "RUNNING", "COMPLETING"]:
                            status = "PENDING" if token == "PD" else "RUNNING" if token == "R" else token
                            return status
                    # Default to RUNNING if in queue but can't parse status
                    return "RUNNING"
                else:
                    return "RUNNING"
            else:
                # Job not in queue - check sacct for final status
                result = subprocess.run(
                    ["sacct", "-j", job_id, "-n", "-o", "State"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    status_line = result.stdout.strip().split('\n')[0].strip()
                    return status_line
                else:
                    # If not in squeue and not in sacct, assume completed
                    return "COMPLETED"
                    
        except Exception as e:
            print(f"Error checking job status: {e}")
            return None
