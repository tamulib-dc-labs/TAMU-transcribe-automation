"""
JSON-based audio downloader for processing audio from HLS URLs.

This module handles downloading audio files from HLS M3U8 streaming URLs
using ffmpeg for conversion.
"""

import os
import subprocess
import re
from typing import List, Dict, Any, Tuple, Optional


class JsonAudioDownloader:
    """Downloads audio files from HLS URLs specified in config JSON."""
    
    def __init__(self, input_dir: str, module_load_command: Optional[str] = None, venv_bin_path: Optional[str] = None):
        """
        Initialize JSON audio downloader.
        
        Args:
            input_dir: Base directory where audio files will be saved
            module_load_command: Optional HPC module load command (e.g., "ml FFmpeg")
            venv_bin_path: Optional path to venv bin directory (to find yt-dlp)
        """
        self.input_dir = input_dir
        self.module_load_command = module_load_command
        self.venv_bin_path = venv_bin_path
    
    @staticmethod
    def parse_audio_url(audio_url: str, name: str) -> Tuple[str, str]:
        """
        Parse audio URL to extract folder name and audio filename.
        
        From URL like:
        https://kaltura-pre.library.tamu.edu/avalon/.../02_00113-a_01-medium.mp4/index.m3u8
        
        Args:
            audio_url: Full HLS URL to audio
            name: Name field from config (used as folder name)
            
        Returns:
            Tuple of (folder_name, audio_filename)
        """
        # Folder name comes from the name field
        folder_name = name
        
        # Extract filename from URL - look for pattern between / and .mp4
        # URL format: .../FILENAME.mp4/index.m3u8
        match = re.search(r'/([^/]+\.mp4)/index\.m3u8', audio_url)
        if match:
            audio_filename = match.group(1)
        else:
            # Fallback: use name with .mp4 extension
            audio_filename = f"{name}.mp4"
        
        return folder_name, audio_filename
    
    def download_audio(self, audio_url: str, dest_folder: str, filename: str) -> bool:
        """
        Download audio from HLS URL using yt-dlp.
        
        Args:
            audio_url: HLS M3U8 URL
            dest_folder: Destination folder path
            filename: Output filename
            
        Returns:
            bool: True if successful
        """
        os.makedirs(dest_folder, exist_ok=True)
        
        output_path = os.path.join(dest_folder, filename)
        
        # Check if file already exists
        if os.path.exists(output_path):
            print(f"  File already exists: {output_path}")
            return True
        
        print(f"  Downloading: {filename}")
        print(f"    From: {audio_url}")
        print(f"    To: {output_path}")
        
        # Use yt-dlp from venv if available, otherwise use system yt-dlp
        if self.venv_bin_path:
            yt_dlp_path = os.path.join(self.venv_bin_path, "yt-dlp")
            cmd = f'"{yt_dlp_path}" --no-warnings -o "{output_path}" "{audio_url}"'
        else:
            cmd = f'yt-dlp --no-warnings -o "{output_path}" "{audio_url}"'
        
        if self.module_load_command:
            cmd = f'{self.module_load_command} && {cmd}'
        
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            print(f"  Downloaded successfully: {filename}")
            return True
        else:
            stderr_msg = result.stderr or ""
            stdout_msg = result.stdout or ""
            error_msg = stderr_msg[-300:] if stderr_msg else stdout_msg[-300:]
            if error_msg:
                print(f"  yt-dlp failed:\n{error_msg}")
            else:
                print(f"  yt-dlp failed (return code: {result.returncode})")
            return False
    
    def process_config_entries(self, entries: List[Dict[str, Any]], max_files: int = 0) -> List[Dict[str, Any]]:
        """
        Download all audio files from config entries.
        
        Args:
            entries: List of config entries with audio, url, vtt, name fields
            max_files: Maximum number of files to process (0 = no limit)
            
        Returns:
            List of successfully processed entries
        """
        successful_entries = []
        
        # Apply limit if specified
        if max_files > 0 and len(entries) > max_files:
            print(f"\\nLimiting to {max_files} files (out of {len(entries)} total)")
            entries = entries[:max_files]
        
        print(f"\\nProcessing {len(entries)} audio files from config...")
        
        for i, entry in enumerate(entries, 1):
            audio_url = entry.get("audio", "")
            name = entry.get("name", "")
            
            if not audio_url or not name:
                print(f"[{i}/{len(entries)}] Skipping invalid entry: missing audio or name")
                continue
            
            print(f"\n[{i}/{len(entries)}] Processing: {name}")
            
            # Parse URL to get folder and filename
            folder_name, audio_filename = self.parse_audio_url(audio_url, name)
            
            # Create destination folder
            dest_folder = os.path.join(self.input_dir, folder_name)
            
            # Download audio
            if self.download_audio(audio_url, dest_folder, audio_filename):
                # Store original entry info for later URL updates
                successful_entry = entry.copy()
                successful_entry["_folder_name"] = folder_name
                successful_entry["_audio_filename"] = audio_filename
                successful_entries.append(successful_entry)
            else:
                print(f"  Failed to download: {name}")
        
        print(f"\nDownload complete: {len(successful_entries)}/{len(entries)} successful")
        return successful_entries
