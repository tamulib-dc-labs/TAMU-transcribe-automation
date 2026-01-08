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
import tempfile
import soundfile as sf
import noisereduce as nr
import numpy as np

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


def clean_audio(audio_path: str, temp_dir: str = None) -> str:
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    
    # Handle MP4 files - extract audio first using ffmpeg
    audio_to_process = audio_path
    extracted_audio_path = None
    
    if audio_path.lower().endswith('.mp4'):
        import subprocess
        extracted_audio_path = os.path.join(temp_dir, f"extracted_{os.path.basename(audio_path)}")
        extracted_audio_path = os.path.splitext(extracted_audio_path)[0] + ".wav"
        
        try:
            # Extract audio from MP4 using ffmpeg
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM format for soundfile compatibility
                "-ar", "16000",  # 16kHz sample rate for Whisper
                "-ac", "1",  # Mono
                extracted_audio_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            audio_to_process = extracted_audio_path
            logger.info(f"Extracted audio from MP4: {os.path.basename(audio_path)}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract audio from MP4: {e.stderr.decode() if e.stderr else 'Unknown error'}")
            raise
    
    data, rate = sf.read(audio_to_process)
    
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    chunk_size = rate * 30
    reduced_noise = np.zeros_like(data)

    for channel in range(data.shape[1]):
        channel_data = data[:, channel]
        num_chunks = int(np.ceil(len(channel_data) / chunk_size))
        
        for chunk_idx in range(num_chunks):
            i = chunk_idx * chunk_size
            end = min(i + chunk_size, len(channel_data))
            chunk = channel_data[i:end]
            
            reduced_chunk = nr.reduce_noise(
                y=chunk, sr=rate, stationary=True,
                n_std_thresh_stationary=2.5, prop_decrease=0.8,
                freq_mask_smooth_hz=1000, time_mask_smooth_ms=100,
                n_fft=2048, hop_length=512, clip_noise_stationary=True,
            )
            reduced_noise[i:end, channel] = reduced_chunk

    boost_factor = 3.0
    boosted_audio = reduced_noise * boost_factor
    max_val = np.abs(boosted_audio).max()
    if max_val > 1.0:
        boosted_audio = boosted_audio / max_val

    if boosted_audio.shape[1] == 1:
        boosted_audio = boosted_audio.squeeze()
    
    cleaned_path = os.path.join(temp_dir, f"cleaned_{os.path.basename(audio_path)}")
    cleaned_path = os.path.splitext(cleaned_path)[0] + ".wav"
    sf.write(cleaned_path, boosted_audio, rate)
    
    # Clean up extracted audio file
    if extracted_audio_path and os.path.exists(extracted_audio_path):
        try:
            os.remove(extracted_audio_path)
        except:
            pass
    
    return cleaned_path


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


def transcribe_with_silero(audio_path: str, model_pipeline, align_model, metadata,
                           device: str, language: str = None) -> Dict:
    cleaned_audio_path = None
    try:
        cleaned_audio_path = clean_audio(audio_path)
        
        # Improved VAD parameters for better alignment
        # Higher min_silence creates longer segments that align better
        vad_params = dict(
            min_silence_duration_ms=700,   # Increased - creates longer segments
            speech_pad_ms=500,              # More padding for context
            min_speech_duration_ms=250      # Skip very short utterances
        )
        
        segments_generator, info = model_pipeline.model.transcribe(
            cleaned_audio_path,
            language=language,
            beam_size=DEFAULT_ASR_OPTIONS["beam_size"],
            vad_filter=True, 
            vad_parameters=vad_params
        )
        
        raw_segments = list(segments_generator)
        
        formatted_segments = []
        for seg in raw_segments:
            formatted_segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text
            })
            
        result = {"segments": formatted_segments, "language": info.language}
        
        if align_model is not None and formatted_segments:
            logger.info(f"Attempting alignment for {len(formatted_segments)} segments...")
            
            # Merge very short segments with adjacent ones for better alignment context
            # Short isolated segments often fail alignment
            MIN_WORDS_FOR_STANDALONE = 3
            merged_segments = []
            
            i = 0
            while i < len(formatted_segments):
                current = formatted_segments[i].copy()
                text = current.get("text", "").strip()
                word_count = len(text.split())
                
                # If short segment, try to merge with next segment
                if word_count < MIN_WORDS_FOR_STANDALONE and i + 1 < len(formatted_segments):
                    next_seg = formatted_segments[i + 1]
                    # Merge if they're close in time (within 2 seconds)
                    time_gap = next_seg.get("start", 0) - current.get("end", 0)
                    if time_gap < 2.0:
                        current["text"] = text + " " + next_seg.get("text", "").strip()
                        current["end"] = next_seg.get("end", current.get("end"))
                        i += 1  # Skip the next segment since we merged it
                
                merged_segments.append(current)
                i += 1
            
            if len(merged_segments) < len(formatted_segments):
                logger.info(f"  Merged {len(formatted_segments) - len(merged_segments)} short segments for better alignment")
            
            try:
                audio_obj = whisperx.load_audio(cleaned_audio_path)
                
                aligned_result = whisperx.align(
                    merged_segments, align_model, metadata, 
                    audio_obj, device, return_char_alignments=False
                )
                
                logger.info(f"  Alignment completed!")
                logger.info(f"  Aligned segments: {len(aligned_result.get('segments', []))}")
                
                # Analyze alignment results and collect statistics
                alignment_stats = {
                    'total_segments': len(formatted_segments),
                    'aligned_segments': 0,
                    'failed_segments': 0,
                    'total_words': 0
                }
                
                for seg in aligned_result.get('segments', []):
                    if 'words' in seg and len(seg['words']) > 0:
                        alignment_stats['aligned_segments'] += 1
                        alignment_stats['total_words'] += len(seg['words'])
                    else:
                        alignment_stats['failed_segments'] += 1
                
                aligned_result["language"] = info.language
                
                # Add alignment statistics to result
                aligned_result["alignment_stats"] = {
                    "total_segments": alignment_stats['total_segments'],
                    "successful_alignments": alignment_stats['aligned_segments'],
                    "failed_alignments": alignment_stats['failed_segments'],
                    "success_rate": round(alignment_stats['aligned_segments'] / max(1, alignment_stats['total_segments']) * 100, 1),
                    "total_words": alignment_stats['total_words']
                }
                
                result = aligned_result
                
                # Log statistics
                success_rate = alignment_stats['aligned_segments'] / max(1, alignment_stats['total_segments']) * 100
                logger.info(f"  ✓ Alignment Statistics:")
                logger.info(f"    - Success rate: {success_rate:.1f}% ({alignment_stats['aligned_segments']}/{alignment_stats['total_segments']} segments)")
                logger.info(f"    - Total words aligned: {alignment_stats['total_words']}")
                logger.info(f"    - Failed alignments: {alignment_stats['failed_segments']}")
                
                if success_rate < 70:
                    logger.warning(f"  ⚠ Low alignment success rate ({success_rate:.1f}%) - consider:")
                    logger.warning(f"    - Checking audio quality")
                    logger.warning(f"    - Adjusting VAD parameters")
                    logger.warning(f"    - Reviewing transcription accuracy")
                elif success_rate >= 90:
                    logger.info(f"  ✓ Excellent alignment quality!")
                    
            except Exception as align_error:
                logger.error(f"  ✗ Alignment FAILED with error: {align_error}")
                import traceback
                logger.error(f"  Traceback: {traceback.format_exc()}")
                # Keep original result without alignment
        elif align_model is None:
            logger.warning(f"Skipping alignment - align_model is None")
        elif not formatted_segments:
            logger.warning(f"Skipping alignment - no segments to align")
        
        return result
        
    except Exception as e:
        logger.error(f"Error transcribing {audio_path}: {str(e)}")
        return None
    finally:
        if cleaned_audio_path and os.path.exists(cleaned_audio_path):
            try:
                os.remove(cleaned_audio_path)
            except:
                pass


def transcribe_directory(input_dir: str, output_dir: str, model_name: str = "large-v2",
                         batch_size: int = 4, compute_type: str = "float16", 
                         language: str = None, align: bool = True, model_dir: str = None, 
                         max_line_width: int = 42, max_line_count: int = 2, 
                         highlight_words: bool = False):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = model_dir if model_dir else model_name
    audio_files = find_audio_files(input_path, output_path)
    
    if not audio_files:
        logger.warning("No audio files found!")
        return
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    model = whisperx.load_model(model_path, device, compute_type=compute_type, language=language if language else "en")
    
    align_model = None
    metadata = None
    if align:
        align_lang = language if language else "en"
        try:
            logger.info(f"Loading alignment model for '{align_lang}' from cache...")
            logger.info(f"  HF_HOME: {os.environ.get('HF_HOME', 'NOT SET')}")
            logger.info(f"  HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE', 'NOT SET')}")
            logger.info(f"  TRANSFORMERS_OFFLINE: {os.environ.get('TRANSFORMERS_OFFLINE', 'NOT SET')}")
            
            align_model, metadata = whisperx.load_align_model(language_code=align_lang, device=device)
            logger.info(f"✓ Successfully loaded alignment model for: {align_lang}")
        except Exception as e:
            logger.error(f"✗ Failed to load alignment model: {str(e)}")
            logger.error(f"  If running on compute node (offline), ensure models were pre-downloaded on login node")
            logger.error(f"  Run on LOGIN NODE: python scripts/check_alignment_cache.py")
            align_model, metadata = None, None

    successful = 0
    failed = 0
    
    for audio_file in tqdm(audio_files, desc="Transcribing"):
        try:
            result = transcribe_with_silero(
                str(audio_file), model, align_model, metadata,
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
    
    align_model = None
    metadata = None
    align_lang = language if language else "en"
    try:
        align_model, metadata = whisperx.load_align_model(language_code=align_lang, device=device)
        logger.info(f"[GPU {gpu_id}] Loaded alignment model for: {align_lang}")
    except Exception as e:
        logger.error(f"[GPU {gpu_id}] Failed to load alignment model: {str(e)}")
    
    for audio_file in file_batch:
        try:
            result = transcribe_with_silero(
                str(audio_file), model, align_model, metadata,
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
                                   compute_type, language, True, model_dir,
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--language", default=None)
    parser.add_argument("--no-align", action="store_true")
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
            align=not args.no_align,
            model_dir=args.model_dir,
            max_line_width=args.max_line_width,
            max_line_count=args.max_line_count,
            highlight_words=args.highlight_words
        )