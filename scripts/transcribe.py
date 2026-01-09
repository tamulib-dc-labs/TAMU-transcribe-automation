import os
import json
import sys

# ============================================================================
# CRITICAL: Set environment variables BEFORE importing whisperx/torch
# This ensures models are loaded from cache in offline mode
# ============================================================================
if 'HF_HOME' in os.environ:
    # Set all cache-related environment variables
    os.environ['TORCH_HOME'] = os.environ['HF_HOME']
    os.environ['HUB_HOME'] = os.path.join(os.environ['HF_HOME'], 'hub')
    os.environ['XDG_CACHE_HOME'] = os.path.dirname(os.environ['HF_HOME'])

# Force offline mode if HF_HUB_OFFLINE is set in environment
# This prevents any network access attempts
if os.environ.get('HF_HUB_OFFLINE') == '1':
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Now safe to import torch and whisperx - they will use cache paths
import torch
import whisperx
from whisperx.utils import get_writer
from pathlib import Path
from typing import List, Dict
import logging
from tqdm import tqdm
import gc
import warnings
import argparse

import nltk
if 'NLTK_DATA' in os.environ:
    nltk.data.path.append(os.environ['NLTK_DATA'])

# Set torch hub directory after import
if 'HF_HOME' in os.environ:
    torch.hub.set_dir(os.environ['HF_HOME'])

try:
    torch.serialization.add_safe_globals = lambda x: None
    import functools
    _original_torch_load = torch.load
    @functools.wraps(_original_torch_load)
    def _patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
except Exception:
    pass

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_ASR_OPTIONS = {"beam_size": 5, "patience": 1.0}


def calculate_word_score_buckets(result: Dict) -> Dict[str, float]:
    """
    Calculate word score bucket thresholds based on percentiles.
    
    Returns thresholds that divide scores into four buckets:
    - Bad: 25th percentile threshold (below = terrible)
    - Neutral: 50th percentile threshold (below = poor)
    - Good: 75th percentile threshold (below = mediocre, above = good)
    """
    all_scores = []
    for segment in result.get('segments', []):
        if 'words' in segment and segment['words']:
            for word in segment['words']:
                score = word.get('score')
                if score is not None:
                    all_scores.append(score)
    
    if not all_scores:
        return {"Good": 0.9, "Neutral": 0.7, "Bad": 0.5}
    
    all_scores.sort()
    n = len(all_scores)
    
    # Calculate percentile-based thresholds
    # 25th percentile = Bad threshold (bottom 25% are terrible)
    # 50th percentile = Neutral threshold (bottom 50% are poor)
    # 75th percentile = Good threshold (top 25% are good)
    idx_25 = int(n * 0.25)
    idx_50 = int(n * 0.50)
    idx_75 = int(n * 0.75)
    
    bad_threshold = round(all_scores[idx_25] if idx_25 < n else all_scores[-1], 3)
    neutral_threshold = round(all_scores[idx_50] if idx_50 < n else all_scores[-1], 3)
    good_threshold = round(all_scores[idx_75] if idx_75 < n else all_scores[-1], 3)
    
    return {
        "Good": good_threshold,
        "Neutral": neutral_threshold,
        "Bad": bad_threshold
    }


def save_transcription(result: Dict, audio_path: Path, output_dir: Path, 
                       max_line_width: int = 42, max_line_count: int = 2, 
                       highlight_words: bool = False, gpu_id: str = None):
    if result is None or 'segments' not in result:
        logger.warning(f"No segments found for {audio_path}")
        return
    
    if "language" not in result:
        result["language"] = "en"
    
    json_output_dir = output_dir / "json"
    vtt_output_dir = output_dir / "vtts"
    json_output_dir.mkdir(parents=True, exist_ok=True)
    vtt_output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = audio_path.stem
    json_path = json_output_dir / f"{base_name}.json"
    
    word_score_buckets = calculate_word_score_buckets(result)
    result_with_buckets = dict(result)
    result_with_buckets["word_score_buckets"] = word_score_buckets
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_with_buckets, f, indent=2, ensure_ascii=False)
    
    try:
        vtt_writer = get_writer("vtt", str(vtt_output_dir))
        vtt_options = {
            "max_line_width": max_line_width,
            "max_line_count": max_line_count,
            "highlight_words": highlight_words
        }
        vtt_writer(result, str(audio_path.with_suffix('')), options=vtt_options)
        logger.info(f"Saved: {base_name}.json/.vtt")
    except Exception as e:
        logger.error(f"Error saving VTT for {audio_path.name}: {str(e)}")


def find_audio_files(root_dir: Path, output_dir: Path = None, 
                     extensions: List[str] = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mp4']) -> List[Path]:
    audio_files = []
    out_str = str(output_dir.resolve()) if output_dir else ""
    for ext in extensions:
        for p in root_dir.rglob(f"*{ext}"):
            if out_str and out_str in str(p.resolve()):
                continue
            audio_files.append(p)
    return sorted(audio_files)


def transcribe_audio(audio_path: str, model, align_model, align_metadata, 
                      device: str, language: str = None) -> Dict:
    """
    Transcription using WhisperX with alignment for word-level timestamps and scores.
    """
    try:
        logger.info(f"Transcribing: {os.path.basename(audio_path)}")
        
        # Load audio using whisperx
        audio = whisperx.load_audio(audio_path)
        
        # Transcribe
        result = model.transcribe(audio, batch_size=16, language=language)
        logger.info(f"  Transcribed {len(result.get('segments', []))} segments")
        
        # Apply alignment for word-level timestamps and scores
        if align_model is not None and result.get('segments'):
            try:
                result = whisperx.align(
                    result["segments"], 
                    align_model, 
                    align_metadata, 
                    audio, 
                    device, 
                    return_char_alignments=False
                )
                logger.info(f"  Aligned with word-level timestamps")
            except Exception as align_error:
                logger.warning(f"  Alignment failed: {align_error}, keeping segment-level only")
        
        return result
        
    except Exception as e:
        logger.error(f"Error transcribing {audio_path}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def transcribe_directory(input_dir: str, output_dir: str, model_name: str = "large-v2",
                         batch_size: int = 16, compute_type: str = "float16", 
                         language: str = None, model_dir: str = None, 
                         max_line_width: int = 42, max_line_count: int = 2, 
                         highlight_words: bool = False):
    """
    Transcribe all audio files in a directory using WhisperX with alignment.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = model_dir if model_dir else model_name
    audio_files = find_audio_files(input_path, output_path)
    
    if not audio_files:
        logger.warning("No audio files found!")
        return
    
    logger.info(f"Found {len(audio_files)} audio files")
    logger.info(f"Using device: {device}")
    logger.info(f"Model: {model_path}")
    
    # Load the whisperx model
    asr_options = {
        "suppress_numerals": True,  # Reduce WER by spelling out numbers
    }
    model = whisperx.load_model(model_path, device, compute_type=compute_type, 
                                 language=language if language else "en",
                                 asr_options=asr_options)
    
    # Load alignment model for word-level timestamps
    align_lang = language if language else "en"
    align_model = None
    align_metadata = None
    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=align_lang, device=device
        )
        logger.info(f"Loaded alignment model for: {align_lang}")
    except Exception as e:
        logger.warning(f"Failed to load alignment model: {e}")
        logger.warning("Will proceed without word-level alignment")

    successful = 0
    failed = 0
    
    for audio_file in tqdm(audio_files, desc="Transcribing"):
        try:
            result = transcribe_audio(
                str(audio_file), model, align_model, align_metadata,
                device, language=language if language else "en"
            )

            if result:
                save_transcription(result, audio_file, output_path, 
                                 max_line_width, max_line_count, highlight_words)
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            logger.error(f"Failed on {audio_file.name}: {e}")
            failed += 1
        
        if (successful + failed) % 5 == 0:
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    
    logger.info(f"Complete! Success: {successful}, Failed: {failed}")
    del model
    if align_model:
        del align_model
    gc.collect()

def _process_gpu_batch(args):
    gpu_id, file_batch, output_path, model_name, model_dir, batch_size, compute_type, language, max_line_width, max_line_count, highlight_words = args
    
    try:
        import functools
        torch.serialization.add_safe_globals = lambda x: None
        _original_torch_load = torch.load
        @functools.wraps(_original_torch_load)
        def _patched_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return _original_torch_load(*args, **kwargs)
        torch.load = _patched_torch_load
    except:
        pass
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = "cuda"
    model_path = model_dir if model_dir else model_name
    
    model = whisperx.load_model(model_path, device, compute_type=compute_type, language=language if language else "en")
    logger.info(f"[GPU {gpu_id}] Loaded model: {model_path}")
    
    # Load alignment model for word-level timestamps
    align_lang = language if language else "en"
    align_model = None
    align_metadata = None
    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=align_lang, device=device
        )
        logger.info(f"[GPU {gpu_id}] Loaded alignment model for: {align_lang}")
    except Exception as e:
        logger.warning(f"[GPU {gpu_id}] Failed to load alignment model: {e}")
    
    for audio_file in file_batch:
        try:
            result = transcribe_audio(
                str(audio_file), model, align_model, align_metadata,
                device, language=language if language else "en"
            )
            if result:
                save_transcription(result, audio_file, output_path, 
                                 max_line_width, max_line_count, highlight_words, str(gpu_id))
        except Exception as e:
            logger.error(f"[GPU {gpu_id}] Failed on {audio_file}: {str(e)}")
    
    del model
    if align_model:
        del align_model
    gc.collect()
    torch.cuda.empty_cache()


def transcribe_directory_parallel(input_dir: str, output_dir: str, model_name: str = "large-v2",
                                  batch_size: int = 4, compute_type: str = "float16", 
                                  language: str = None, num_gpus: int = None, model_dir: str = None, 
                                  max_line_width: int = 42, max_line_count: int = 2, 
                                  highlight_words: bool = False):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if available_gpus == 0:
        logger.warning("No GPUs available, falling back to sequential")
        return transcribe_directory(input_dir, output_dir, model_name, batch_size, 
                                   compute_type, language, model_dir,
                                   max_line_width, max_line_count, highlight_words)
    
    if num_gpus is None or num_gpus > available_gpus:
        num_gpus = available_gpus
    
    logger.info(f"Using {num_gpus} GPUs for parallel processing")
    
    audio_files = find_audio_files(input_path, output_path)
    logger.info(f"Found {len(audio_files)} audio files")
    
    files_per_gpu = (len(audio_files) + num_gpus - 1) // num_gpus
    
    gpu_batches = []
    for i in range(num_gpus):
        start_idx = i * files_per_gpu
        end_idx = min((i + 1) * files_per_gpu, len(audio_files))
        if start_idx < len(audio_files):
            batch_args = (i, audio_files[start_idx:end_idx], output_path, model_name, 
                         model_dir, batch_size, compute_type, language,
                         max_line_width, max_line_count, highlight_words)
            gpu_batches.append(batch_args)
    
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        list(tqdm(executor.map(_process_gpu_batch, gpu_batches), total=len(gpu_batches), desc="GPU workers"))
    
    logger.info("All GPUs finished processing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--model", default="large-v2")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--language", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--max-line-width", type=int, default=42)
    parser.add_argument("--max-line-count", type=int, default=2)
    parser.add_argument("--highlight-words", action="store_true")
    parser.add_argument("--parallel", action="store_true", help="Use multiple GPUs in parallel")
    parser.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs to use")
    
    args = parser.parse_args()
    
    if args.parallel and torch.cuda.device_count() > 1:
        transcribe_directory_parallel(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_name=args.model,
            batch_size=args.batch_size,
            compute_type=args.compute_type,
            language=args.language,
            num_gpus=args.num_gpus,
            model_dir=args.model_dir,
            max_line_width=args.max_line_width,
            max_line_count=args.max_line_count,
            highlight_words=args.highlight_words
        )
    else:
        transcribe_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_name=args.model,
            batch_size=args.batch_size,
            compute_type=args.compute_type,
            language=args.language,
            model_dir=args.model_dir,
            max_line_width=args.max_line_width,
            max_line_count=args.max_line_count,
            highlight_words=args.highlight_words
        )