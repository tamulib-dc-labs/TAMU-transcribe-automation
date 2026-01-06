"""
Model downloader for WhisperX models and alignment models.
"""

import os
import sys
from typing import List, Optional


class ModelDownloader:
    """Downloads WhisperX models for offline use."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize model downloader.
        
        Args:
            cache_dir: HuggingFace cache directory
        """
        self.cache_dir = cache_dir
    
    def _apply_pytorch_patch(self):
        """Apply PyTorch 2.6+ compatibility patch for weights_only issue."""
        import torch
        
        try:
            torch.serialization.add_safe_globals = lambda x: None
            import functools
            _original_torch_load = torch.load
            
            @functools.wraps(_original_torch_load)
            def _patched_torch_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return _original_torch_load(*args, **kwargs)
            
            torch.load = _patched_torch_load
            print("PyTorch 2.6+ compatibility patch applied: weights_only=False for all torch.load calls")
            sys.stdout.flush()
        except Exception as e:
            print(f"WARNING: Failed to apply PyTorch compatibility patch: {e}")
            print("Model loading may fail on CPU nodes with PyTorch 2.6+")
            sys.stdout.flush()
    
    def download_models(
        self,
        model_name: str = "large-v3",
        languages: Optional[List[str]] = None,
        compute_type: str = "float16"
    ) -> bool:
        """
        Download WhisperX models and alignment models.
        
        Args:
            model_name: WhisperX model to download
            languages: List of language codes for alignment models
            compute_type: Compute type for the model
            
        Returns:
            bool: True if successful
        """
        # Import torch and whisperx only when needed
        import torch
        import whisperx
        
        # Apply PyTorch compatibility patch before using models
        self._apply_pytorch_patch()
        
        # Set HF cache if provided
        if self.cache_dir:
            os.environ['HF_HOME'] = self.cache_dir
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"Using cache directory: {self.cache_dir}")
        
        # Set offline mode to False for downloading
        os.environ['HF_HUB_OFFLINE'] = '0'
        
        # Auto-detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Adjust compute type based on device
        original_compute_type = compute_type
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"
            print(f"Warning: CPU detected, float16 not supported. Switching from {original_compute_type} to {compute_type}.")
        
        print(f"Using device: {device}, compute_type: {compute_type}")
        
        if languages is None:
            languages = ["en", "es", "fr", "de"]
        
        cache_location = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
        
        print(f"Downloading WhisperX model: {model_name}...")
        print("This may take several minutes for large models...")
        print(f"Model will be cached at: {cache_location}")
        sys.stdout.flush()
        
        try:
            model = whisperx.load_model(model_name, device, compute_type=compute_type)
            print(f"WhisperX model '{model_name}' downloaded successfully!")
            sys.stdout.flush()
            del model
        except Exception as e:
            print(f"Error downloading WhisperX model: {e}")
            sys.stdout.flush()
            return False
        
        print(f"\nDownloading alignment models for languages: {', '.join(languages)}...")
        sys.stdout.flush()
        for lang in languages:
            try:
                print(f"  Downloading alignment model for '{lang}'...")
                sys.stdout.flush()
                align_model, metadata = whisperx.load_align_model(language_code=lang, device=device)
                print(f"  Alignment model for '{lang}' downloaded!")
                sys.stdout.flush()
                del align_model
            except Exception as e:
                print(f"  Could not download alignment for '{lang}': {e}")
                sys.stdout.flush()
        
        print(f"\n{'='*60}")
        print(f"All models downloaded to: {cache_location}")
        print(f"{'='*60}")
        
        # CRITICAL: Ensure alignment models are in checkpoints/ subdirectory
        # Torchaudio expects models at: {HF_HOME}/checkpoints/model_name.pth
        print(f"\nVerifying alignment model structure for offline mode...")
        checkpoints_dir = os.path.join(cache_location, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        # Check for alignment model files
        alignment_patterns = ["wav2vec2_*.pth", "wav2vec2_*.pt"]
        found_models = []
        
        for pattern in alignment_patterns:
            import glob
            models = glob.glob(os.path.join(cache_location, pattern))
            for model_file in models:
                model_name = os.path.basename(model_file)
                checkpoint_path = os.path.join(checkpoints_dir, model_name)
                
                if not os.path.exists(checkpoint_path):
                    import shutil
                    shutil.copy2(model_file, checkpoint_path)
                    print(f"  ✓ Copied {model_name} to checkpoints/")
                    found_models.append(model_name)
                else:
                    print(f"  ✓ {model_name} already in checkpoints/")
                    found_models.append(model_name)
        
        if found_models:
            print(f"\nAlignment models ready for offline use: {len(found_models)} models")
        else:
            print(f"\n⚠ WARNING: No alignment models found. Run:")
            print(f"  export HF_HOME={cache_location}")
            print(f"  python -c 'import whisperx; whisperx.load_align_model(\"en\", \"cpu\")'")
        
        return True
