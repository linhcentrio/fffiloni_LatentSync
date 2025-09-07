#!/usr/bin/env python3
"""
RunPod Serverless Handler for LatentSync v1.0
Optimized version - Core functionality only (ffmpeg-based, no moviepy)
"""

import runpod
import os
import tempfile
import uuid
import requests
import time
import torch
import sys
import shutil
import subprocess
import json
from pathlib import Path
from minio import Minio
from urllib.parse import quote
from omegaconf import OmegaConf
import logging
import gc
from datetime import datetime

# Add path for local modules
sys.path.append('/app')

# Import required modules
try:
    from diffusers import AutoencoderKL, DDIMScheduler
    from latentsync.models.unet import UNet3DConditionModel
    from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
    from accelerate.utils import set_seed
    from latentsync.whisper.audio2feature import Audio2Feature
except ImportError as e:
    logging.error(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configurations for LatentSync v1.0
MODEL_CONFIG = {
    "config_path": "/app/configs/unet/second_stage.yaml",
    "checkpoint_path": "/app/checkpoints/latentsync_unet.pt",
    "whisper_tiny_path": "/app/checkpoints/whisper/tiny.pt",
    "whisper_small_path": "/app/checkpoints/whisper/small.pt",
    "default_guidance_scale": 1.0
}

# MinIO Configuration
MINIO_ENDPOINT = "media.aiclip.ai"
MINIO_ACCESS_KEY = "VtZ6MUPfyTOH3qSiohA2"
MINIO_SECRET_KEY = "8boVPVIynLEKcgXirrcePxvjSk7gReIDD9pwto3t"
MINIO_BUCKET = "video"
MINIO_SECURE = False

# Initialize MinIO client
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE
)

# Global model instances
config = None
pipeline = None

def get_media_info(file_path: str) -> dict:
    """Get media file information using ffprobe (replaces moviepy/pydub)"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            logger.error(f"ffprobe failed: {result.stderr}")
            return {}
            
        info = json.loads(result.stdout)
        return info
        
    except Exception as e:
        logger.error(f"Failed to get media info: {e}")
        return {}

def get_duration(file_path: str) -> float:
    """Get media duration using ffprobe (replaces get_video_duration/get_audio_duration)"""
    try:
        info = get_media_info(file_path)
        duration = float(info.get('format', {}).get('duration', 0))
        return duration
    except Exception as e:
        logger.error(f"Failed to get duration: {e}")
        return 0.0

def crop_video_ffmpeg(video_path: str, temp_dir: str, max_duration: int = 10) -> str:
    """Crop video using ffmpeg (replaces moviepy crop)"""
    try:
        input_file_name = os.path.basename(video_path)
        output_path = os.path.join(temp_dir, f"cropped_{input_file_name}")
        
        duration = get_duration(video_path)
        
        if duration > max_duration:
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-t', str(max_duration),
                '-c:v', 'libx264', '-c:a', 'aac',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"✂️ Video cropped: {duration:.1f}s → {max_duration}s")
                return output_path
            else:
                logger.error(f"ffmpeg crop failed: {result.stderr}")
                return video_path
        else:
            # If already shorter, just copy
            shutil.copy2(video_path, output_path)
            return output_path
            
    except Exception as e:
        logger.error(f"Failed to crop video: {e}")
        return video_path

def create_reverse_video_ffmpeg(video_path: str, temp_dir: str) -> str:
    """Create reverse video using ffmpeg (replaces moviepy reverse)"""
    try:
        input_file_name = os.path.basename(video_path)
        output_path = os.path.join(temp_dir, f"reverse_{input_file_name}")
        
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vf', 'reverse',
            '-af', 'areverse',
            '-c:v', 'libx264', '-c:a', 'aac',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            return output_path
        else:
            logger.error(f"ffmpeg reverse failed: {result.stderr}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to create reverse video: {e}")
        return None

def create_pingpong_loop_ffmpeg(video_path: str, temp_dir: str, target_duration: float) -> str:
    """Create pingpong loop using ffmpeg (replaces moviepy concatenation)"""
    try:
        video_duration = get_duration(video_path)
        
        if video_duration <= 0:
            logger.error("Invalid video duration")
            return video_path
        
        # Create reverse video
        reverse_path = create_reverse_video_ffmpeg(video_path, temp_dir)
        if not reverse_path:
            logger.warning("Failed to create reverse video, using original")
            return video_path
        
        # Calculate how many loops we need
        cycle_duration = video_duration * 2  # forward + reverse
        num_full_cycles = int(target_duration // cycle_duration)
        remaining_time = target_duration % cycle_duration
        
        # Create concat file list
        concat_file = os.path.join(temp_dir, "concat_list.txt")
        input_file_name = os.path.basename(video_path)
        output_path = os.path.join(temp_dir, f"pingpong_{input_file_name}")
        
        with open(concat_file, 'w') as f:
            # Add full cycles
            for _ in range(num_full_cycles):
                f.write(f"file '{video_path}'\n")
                f.write(f"file '{reverse_path}'\n")
            
            # Add remaining partial cycle
            if remaining_time > 0:
                if remaining_time <= video_duration:
                    # Only partial forward
                    f.write(f"file '{video_path}'\n")
                else:
                    # Full forward + partial reverse
                    f.write(f"file '{video_path}'\n")
                    f.write(f"file '{reverse_path}'\n")
        
        # Create the final concatenated video
        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', concat_file,
            '-t', str(target_duration),
            '-c:v', 'libx264', '-c:a', 'aac',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            logger.info(f"🔄 Pingpong loop created: {video_duration:.1f}s → {target_duration:.1f}s")
            
            # Cleanup
            if os.path.exists(reverse_path):
                os.remove(reverse_path)
            if os.path.exists(concat_file):
                os.remove(concat_file)
                
            return output_path
        else:
            logger.error(f"ffmpeg concat failed: {result.stderr}")
            return video_path
            
    except Exception as e:
        logger.error(f"Failed to create pingpong loop: {e}")
        return video_path

def simplified_process_inputs_ffmpeg(video_path: str, audio_path: str, crop_inputs: bool = False, pingpong: bool = True):
    """
    Simplified processing using ffmpeg instead of moviepy:
    - crop_inputs: Whether to crop video to 10s max
    - pingpong: Whether to apply pingpong loop when audio > video
    """
    processed_video_path = video_path
    processed_audio_path = audio_path
    temp_dir = tempfile.mkdtemp()
    
    # Step 1: Crop logic (independent)
    if crop_inputs:
        video_duration = get_duration(video_path)
        if video_duration > 10.0:
            processed_video_path = crop_video_ffmpeg(video_path, temp_dir, max_duration=10)
            logger.info(f"✂️ Video cropped: {video_duration:.1f}s → 10.0s")
        else:
            logger.info(f"✅ Video already ≤ 10s: {video_duration:.1f}s")
    else:
        video_duration = get_duration(video_path)
        logger.info(f"🎬 Keeping original video duration: {video_duration:.1f}s")
    
    # Step 2: Pingpong logic (independent)
    if pingpong:
        final_video_duration = get_duration(processed_video_path)
        audio_duration = get_duration(audio_path)
        
        logger.info(f"📊 Duration check: Video={final_video_duration:.1f}s, Audio={audio_duration:.1f}s")
        
        if audio_duration > final_video_duration:
            processed_video_path = create_pingpong_loop_ffmpeg(
                processed_video_path, temp_dir, 
                target_duration=audio_duration
            )
            logger.info(f"🔄 Pingpong loop applied: {final_video_duration:.1f}s → {audio_duration:.1f}s")
        else:
            logger.info(f"✅ No loop needed: Video({final_video_duration:.1f}s) ≥ Audio({audio_duration:.1f}s)")
    else:
        logger.info("🎯 Using LatentSync native duration handling")
    
    return processed_video_path, processed_audio_path

def initialize_models():
    """Initialize LatentSync v1.0 pipeline"""
    global config, pipeline
    
    try:
        # Load LatentSync v1.0 configuration
        logger.info("🚀 Loading LatentSync v1.0 configuration...")
        config_path = Path(MODEL_CONFIG["config_path"])
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        config = OmegaConf.load(config_path)
        logger.info(f"✅ Configuration loaded from {config_path}")
        
        # Initialize scheduler
        scheduler = DDIMScheduler.from_pretrained("configs")
        
        # Determine whisper model path based on config
        if config.model.cross_attention_dim == 768:
            whisper_model_path = MODEL_CONFIG["whisper_small_path"]
        elif config.model.cross_attention_dim == 384:
            whisper_model_path = MODEL_CONFIG["whisper_tiny_path"]
        else:
            raise NotImplementedError("cross_attention_dim must be 768 or 384")
        
        # Initialize audio encoder
        audio_encoder = Audio2Feature(
            model_path=whisper_model_path, 
            device="cuda", 
            num_frames=config.data.num_frames
        )
        logger.info(f"✅ Audio encoder initialized with {whisper_model_path}")
        
        # Initialize VAE
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        logger.info("✅ VAE initialized")
        
        # Initialize UNet
        checkpoint_path = Path(MODEL_CONFIG["checkpoint_path"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"UNet checkpoint not found: {checkpoint_path}")
        
        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(config.model),
            str(checkpoint_path),
            device="cpu",
        )
        unet = unet.to(dtype=torch.float16)
        logger.info(f"✅ UNet loaded from {checkpoint_path}")
        
        # Initialize LatentSync pipeline
        pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        ).to("cuda")
        logger.info("✅ LatentSync v1.0 pipeline initialized")

    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        raise e

def download_file(url: str, local_path: str) -> bool:
    """Download file from URL with progress tracking"""
    try:
        logger.info(f"📥 Downloading {url}")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        logger.info(f"✅ Downloaded: {local_path} ({downloaded/1024/1024:.1f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False

def upload_to_minio(local_path: str, object_name: str) -> str:
    """Upload file to MinIO with enhanced error handling"""
    try:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        minio_client.fput_object(MINIO_BUCKET, object_name, local_path)
        file_url = f"https://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
        logger.info(f"✅ Uploaded successfully: {file_url}")
        return file_url
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise e

def run_lipsync_inference(video_path: str, audio_path: str, output_path: str,
                         inference_steps: int, guidance_scale: float, 
                         seed: int, crop_inputs: bool = False, pingpong: bool = True) -> bool:
    """Run LatentSync v1.0 inference"""
    global config, pipeline
    
    try:
        logger.info(f"🎯 Running LatentSync v1.0 inference...")
        
        temp_dir = None
        processed_video_path = video_path
        processed_audio_path = audio_path
        
        # Process inputs using simplified logic
        if crop_inputs or pingpong:
            temp_dir = tempfile.mkdtemp()
            processed_video_path, processed_audio_path = simplified_process_inputs_ffmpeg(
                video_path, audio_path, crop_inputs, pingpong
            )
        else:
            # Both false → Pure LatentSync native (no preprocessing)
            processed_video_path = video_path
            processed_audio_path = audio_path
            logger.info("🎯 Pure LatentSync native processing - no preprocessing")
        
        # Set seed
        if seed != -1:
            set_seed(seed)
        else:
            torch.seed()
        
        logger.info(f"Initial seed: {torch.initial_seed()}")
        
        # Run inference
        unique_id = str(uuid.uuid4())
        video_out_temp_path = f"video_out_{unique_id}.mp4"
        
        pipeline(
            video_path=processed_video_path,
            audio_path=processed_audio_path,
            video_out_path=video_out_temp_path,
            video_mask_path=video_out_temp_path.replace(".mp4", "_mask.mp4"),
            num_frames=config.data.num_frames,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            weight_dtype=torch.float16,
            width=config.data.resolution,
            height=config.data.resolution,
        )
        
        # Move output to final path
        if os.path.exists(video_out_temp_path):
            shutil.move(video_out_temp_path, output_path)
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"✅ LatentSync v1.0 inference completed successfully ({file_size:.1f} MB)")
            
            # Clean up temporary files
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Temporary directory {temp_dir} deleted")
            
            # Clean up mask file if exists
            mask_path = video_out_temp_path.replace(".mp4", "_mask.mp4")
            if os.path.exists(mask_path):
                os.remove(mask_path)
            
            return True
        else:
            logger.error(f"❌ LatentSync v1.0 inference failed - no output file")
            return False
            
    except Exception as e:
        logger.error(f"❌ LatentSync v1.0 inference error: {e}")
        return False

def handler(job):
    """Main RunPod handler for LatentSync v1.0 processing"""
    job_id = job.get("id", "unknown")
    start_time = time.time()
    
    try:
        job_input = job.get("input", {})
        video_url = job_input.get("video_url")
        audio_url = job_input.get("audio_url")
        
        if not video_url or not audio_url:
            return {"error": "Missing video_url or audio_url"}
        
        # Processing parameters
        inference_steps = job_input.get("inference_steps", 20)
        guidance_scale = job_input.get("guidance_scale", MODEL_CONFIG["default_guidance_scale"])
        seed = job_input.get("seed", 1247)
        crop_inputs = job_input.get("crop_inputs", False)  # Default: Keep original video duration
        pingpong = job_input.get("pingpong", True)         # Default: Use pingpong loop when audio > video
        
        logger.info(f"🚀 Job {job_id}: LatentSync v1.0 Video Processing")
        logger.info(f"📺 Video: {video_url}")
        logger.info(f"🎵 Audio: {audio_url}")
        logger.info(f"⚙️ Parameters: steps={inference_steps}, scale={guidance_scale}, seed={seed}, crop={crop_inputs}, pingpong={pingpong}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "input_video.mp4")
            audio_path = os.path.join(temp_dir, "input_audio.wav")
            
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(temp_dir, f"lipsync_v1_{current_time}.mp4")
            
            # Step 1: Download input files
            logger.info("📥 Step 1/3: Downloading input files...")
            if not download_file(video_url, video_path):
                return {"error": "Failed to download video"}
            
            if not download_file(audio_url, audio_path):
                return {"error": "Failed to download audio"}
            
            # Step 2: Run LatentSync v1.0 inference
            logger.info(f"🎯 Step 2/3: Running LatentSync v1.0 processing...")
            lipsync_success = run_lipsync_inference(
                video_path, audio_path, output_path, 
                inference_steps, guidance_scale, seed, crop_inputs, pingpong
            )
            
            if not lipsync_success:
                return {"error": "LatentSync v1.0 processing failed"}
            
            if not os.path.exists(output_path):
                return {"error": "Lipsync output not generated"}
            
            # Step 3: Upload result
            logger.info("📤 Step 3/3: Uploading result...")
            output_filename = f"latentsync_v1_{job_id}_{uuid.uuid4().hex[:8]}.mp4"
            output_url = upload_to_minio(output_path, output_filename)
            
            processing_time = time.time() - start_time
            
            # Prepare response
            response = {
                "output_video_url": output_url,
                "processing_time_seconds": round(processing_time, 2),
                "video_cropped": crop_inputs,
                "pingpong_applied": pingpong,
                "latentsync_version": "1.0",
                "status": "completed"
            }
            
            return response
            
    except Exception as e:
        logger.error(f"❌ Handler error: {e}")
        return {
            "error": str(e),
            "status": "failed",
            "processing_time_seconds": round(time.time() - start_time, 2)
        }
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    logger.info("🚀 Starting LatentSync v1.0 Serverless Worker...")
    logger.info(f"🗄️ Storage: {MINIO_ENDPOINT}/{MINIO_BUCKET}")
    logger.info(f"🤖 Model: LatentSync v1.0 (Core Only)")
    
    # Verify environment
    try:
        logger.info(f"🐍 Python: {sys.version}")
        logger.info(f"🔥 PyTorch: {torch.__version__}")
        logger.info(f"⚡ CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"🎮 GPU: {torch.cuda.get_device_name()}")
            logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    except Exception as e:
        logger.warning(f"⚠️ Environment check failed: {e}")
    
    # Verify model files exist
    config_path = Path(MODEL_CONFIG["config_path"])
    checkpoint_path = Path(MODEL_CONFIG["checkpoint_path"])
    
    if config_path.exists() and checkpoint_path.exists():
        logger.info(f"✅ LatentSync v1.0 files verified")
    else:
        logger.warning(f"⚠️ LatentSync v1.0 files missing")
    
    # Initialize models
    try:
        initialize_models()
        logger.info("✅ All models initialized successfully")
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        sys.exit(1)
    
    # Start RunPod serverless worker
    logger.info("🎬 Ready to process LatentSync v1.0 requests...")
    runpod.serverless.start({"handler": handler})
