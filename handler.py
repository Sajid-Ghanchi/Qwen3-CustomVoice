#Bismillah
"""
RunPod Serverless Handler for Qwen3-TTS-12Hz-1.7B-CustomVoice
"""

import runpod
import base64
import io
import torch
import soundfile as sf
import numpy as np
import traceback

from qwen_tts import Qwen3TTSModel

# Global model instance
tts_model = None

def load_model():
    """Load Qwen3 TTS model with hardware optimizations."""
    global tts_model

    if tts_model is not None:
        return tts_model

    print("[Handler] Loading Qwen3 TTS model...")

    try:
        tts_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        print("[Handler] Model loaded successfully")
    except Exception as e:
        print(f"[Handler] Failed to load model: {e}")
        raise

    return tts_model

def audio_to_base64(audio_array: np.ndarray, sample_rate: int) -> str:
    """Convert audio array to base64 encoded WAV."""
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format="WAV")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def handler(job: dict) -> dict:
    """Main RunPod handler function."""
    
    job_input = job.get("input", {})

    # Health check
    if job_input.get("health_check"):
        return {
            "status": "healthy",
            "message": "Qwen TTS handler ready",
            "model_loaded": tts_model is not None
        }

    # Extract required inputs
    text = job_input.get("text", "")
    if not text:
        return {"error": "No text provided"}

    # Map parameters to Qwen's expected API
    language = job_input.get("language", "Auto")
    speaker = job_input.get("speaker", "Vivian")
    
    # Map legacy emotion input to the new instruct format
    instruct = job_input.get("instruct", job_input.get("emotion", ""))

    try:
        model = load_model()

        print(f"[Handler] Generating speech for: {text[:50]}...")
        print(f"[Handler] Speaker: {speaker}, Language: {language}, Instruct: {instruct}")

        # Execute single inference
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct if instruct else None
        )

        audio = wavs[0]

        # Convert to numpy if tensor
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()

        # Ensure 1D and Normalize
        if len(audio.shape) > 1:
            audio = audio.squeeze()
        audio = audio / np.max(np.abs(audio)) * 0.95

        # Convert to base64
        audio_b64 = audio_to_base64(audio, sample_rate=sr)

        return {
            "audio_base64": audio_b64,
            "sample_rate": sr,
            "duration_seconds": len(audio) / sr,
            "text": text,
            "speaker": speaker,
            "language": language,
            "instruct": instruct
        }

    except torch.cuda.OutOfMemoryError:
        # Gracefully handle OOM errors and clear cache
        torch.cuda.empty_cache()
        return {
            "error": "CUDA Out of Memory. The requested text might be too long for the allocated GPU memory.",
            "traceback": traceback.format_exc()
        }
    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Pre-load model on worker start if GPU is available
print("[Handler] Attempting to pre-load model...")
try:
    if torch.cuda.is_available():
        load_model()
    else:
        print("[Handler] CUDA not available, skipping preload.")
except Exception as e:
    print(f"[Handler] Warning: Could not pre-load model: {e}")

# RunPod serverless entry point
runpod.serverless.start({"handler": handler})
