import os
import sys
import torch
import logging
from pathlib import Path
import numpy as np
import torch.nn.functional as F
import imageio
import gc
import copy

# --- MAC MEMORY FIX (CRITICAL FOR CRASHES) ---
# Disables the artificial memory limit for MPS. 
# Without this, the Mac kills the process even if it has free RAM (Swap).
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# --- MEMORY HELPER ---
def force_cleanup(tensors_to_delete=None):
    """
    Aggressive memory cleanup for MPS/CUDA.
    
    For MPS: synchronize() is CRITICAL before empty_cache() - without it,
    queued GPU operations hold references and memory is never freed.
    """
    # 1. Delete explicit tensors if provided
    if tensors_to_delete:
        for t in tensors_to_delete:
            if t is not None:
                del t
    
    # 2. First garbage collection pass
    gc.collect()
    
    # 3. GPU-specific cleanup
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for all CUDA ops
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        # CRITICAL: synchronize MUST happen before empty_cache on MPS
        # Otherwise queued operations keep memory allocated
        torch.mps.synchronize()
        torch.mps.empty_cache()
    
    # 4. Second garbage collection pass (catches circular refs)
    gc.collect() 

# Add ml-sharp src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_sharp_src = os.path.join(current_dir, "ml-sharp", "src")
if ml_sharp_src not in sys.path:
    sys.path.append(ml_sharp_src)

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils.gaussians import save_ply, SceneMetaData, unproject_gaussians
from sharp.cli.render import render_gaussians
from sharp.cli import render as sharp_render
from sharp.utils import camera
from sharp.utils import gsplat
from sharp.utils import vis  # Required for depth colorization

LOGGER = logging.getLogger(__name__)

# =============================================================================
# HARDWARE ACCELERATED VIDEO WRITING
# Helper to get the best FFMPEG codec based on available hardware.
# =============================================================================
def get_best_video_writer(path, fps=30):
    """
    Returns an imageio writer configured with hardware acceleration if available.
    """
    codec_priority = []
    
    if torch.cuda.is_available() and not sys.platform.startswith("linux"):
        codec_priority.append('h264_nvenc')
    elif torch.backends.mps.is_available():
        codec_priority.append('h264_videotoolbox')
    
    codec_priority.append('libx264')
    pixel_format = 'yuv420p'
    
    for codec in codec_priority:
        try:
            LOGGER.info(f"Attempting video writer with codec: {codec}")
            output_params = []
            
            if codec == 'h264_nvenc':
                output_params = ['-preset', 'p4', '-rc', 'constqp', '-qp', '23']
            elif codec == 'h264_videotoolbox':
                output_params = ['-q:v', '85', '-b:v', '12M', '-allow_sw', '1']
            else:
                output_params = ['-preset', 'ultrafast', '-crf', '23']

            writer = imageio.get_writer(
                path, 
                fps=fps, 
                codec=codec, 
                quality=None,
                macro_block_size=16,
                pixelformat=pixel_format,
                ffmpeg_params=output_params,
                format='FFMPEG'
            )
            return writer
        except Exception as e:
            LOGGER.warning(f"Codec {codec} failed: {e}. Trying next...")
            continue
            
    raise RuntimeError("Could not initialize video writer with any known codec.")

# =============================================================================
# MPS/CUDA COMPATIBILITY: Patched render_gaussians
# This replaces the original function completely to use our custom writers.
# =============================================================================
def render_gaussians_mps_compatible(gaussians, metadata, output_path, params=None, stop_event=None):
    """
    Memory-optimized rendering:
    1. Detaches all tensors from computation graph
    2. Immediate cleanup after each frame write
    3. Aggressive garbage collection every N frames
    """
    # DEBUG: Memory at START of render call
    import psutil
    process = psutil.Process()
    rss_gb = process.memory_info().rss / 1024**3
    traj_name = params.type if params else "default"
    
    mps_info = ""
    if torch.backends.mps.is_available():
        try:
            mps_alloc = torch.mps.driver_allocated_memory() / 1024**3
            mps_info = f", MPS Driver: {mps_alloc:.1f}GB"
        except:
            pass
    
    print(f"=== RENDER START [{traj_name}]: Process RSS={rss_gb:.1f}GB{mps_info} ===", flush=True)
    
    force_cleanup()

    (width, height) = metadata.resolution_px
    f_px = metadata.focal_length_px
    
    render_width = int(width)
    render_height = int(height)

    if params is None:
        params = camera.TrajectoryParams()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        raise RuntimeError("Rendering requires CUDA or MPS.")

    # Scaled intrinsics matrix
    intrinsics_scaled = torch.tensor(
        [
            [f_px, 0, (render_width - 1) / 2.0, 0],
            [0, f_px, (render_height - 1) / 2.0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device=device,
        dtype=torch.float32,
    )
    
    path_color = str(output_path)
    path_depth = str(output_path).replace(".mp4", ".depth.mp4")
    
    writer_color = get_best_video_writer(path_color, fps=30)
    writer_depth = get_best_video_writer(path_depth, fps=30)

    max_depth_estimate = None
    metric_depth_max = 50.0 

    # Track objects for cleanup
    detached_data = None
    gaussians_render = None
    camera_model = None
    trajectory = None
    renderer = None
    intrinsics_batch = None
    
    with torch.no_grad():
        # Detach gaussians completely from computation graph
        detached_data = {}
        for key in gaussians._fields:
            val = getattr(gaussians, key)
            if isinstance(val, torch.Tensor):
                # Critical: detach() + clone() + to(device) breaks all references
                detached_data[key] = val.detach().clone().to(device)
            else:
                detached_data[key] = val
        
        gaussians_render = gaussians.__class__(**detached_data)
        
        # Clear detached_data immediately after creating gaussians_render
        # This breaks the reference chain
        for key in list(detached_data.keys()):
            detached_data[key] = None
        detached_data.clear()
        del detached_data
        detached_data = None
        
        # Pre-compute trajectory (avoids recreation per frame)
        camera_model = camera.create_camera_model(
            gaussians_render, intrinsics_scaled, resolution_px=metadata.resolution_px
        )
        trajectory = camera.create_eye_trajectory(
            gaussians_render, params, resolution_px=metadata.resolution_px, f_px=f_px
        )
        
        # Create renderer ONCE and reuse
        renderer = gsplat.GSplatRenderer(color_space=metadata.color_space)
        intrinsics_batch = intrinsics_scaled[None]

        try:
            num_frames = len(trajectory)
            
            for frame_idx, eye_position in enumerate(trajectory):
                # Check cancellation
                if stop_event and stop_event.is_set():
                    print(f"Render interruped at frame {frame_idx}/{num_frames} by user.")
                    break

                # COMPREHENSIVE MEMORY DEBUG
                if frame_idx % 30 == 0:
                    import psutil
                    process = psutil.Process()
                    rss_gb = process.memory_info().rss / 1024**3
                    vms_gb = process.memory_info().vms / 1024**3
                    sys_mem = psutil.virtual_memory()
                    sys_used_gb = sys_mem.used / 1024**3
                    sys_percent = sys_mem.percent
                    
                    # GPU memory info
                    gpu_info = ""
                    if torch.cuda.is_available():
                        gpu_alloc = torch.cuda.memory_allocated() / 1024**3
                        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
                        gpu_info = f" | CUDA: {gpu_alloc:.1f}GB alloc, {gpu_reserved:.1f}GB reserved"
                    elif torch.backends.mps.is_available():
                        # MPS doesn't have direct memory query, but we can check driver allocated
                        try:
                            mps_alloc = torch.mps.driver_allocated_memory() / 1024**3
                            gpu_info = f" | MPS Driver: {mps_alloc:.1f}GB"
                        except:
                            gpu_info = " | MPS: N/A"
                    
                    print(f"Frame {frame_idx}: Process RSS={rss_gb:.1f}GB, VMS={vms_gb:.1f}GB | System RAM: {sys_used_gb:.1f}GB ({sys_percent}%){gpu_info}", flush=True)
                
                # Compute camera info
                camera_info = camera_model.compute(eye_position)
                
                # Render frame
                rendering_output = renderer(
                    gaussians_render, 
                    extrinsics=camera_info.extrinsics[None].to(device),
                    intrinsics=intrinsics_batch,
                    image_width=render_width,
                    image_height=render_height,
                )
                
                # Process COLOR: immediate numpy conversion and write
                with torch.no_grad():
                    color_tensor = rendering_output.color[0].detach()
                    color_np = (color_tensor.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                    writer_color.append_data(color_np)
                
                # Delete color immediately after write
                del color_tensor, color_np
                
                # Process DEPTH: immediate colorization and write
                with torch.no_grad():
                    depth_tensor = rendering_output.depth[0].detach()
                    
                    if max_depth_estimate is None:
                        max_depth_estimate = depth_tensor.max().item()
                    
                    colored_depth_pt = vis.colorize_depth(
                        depth_tensor,
                        val_max=min(max_depth_estimate, metric_depth_max)
                    )
                    depth_np = colored_depth_pt.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    writer_depth.append_data(depth_np)
                
                # Delete depth immediately after write
                del depth_tensor, colored_depth_pt, depth_np
                
                # Critical: delete rendering_output and camera_info to break references
                del rendering_output
                del camera_info
                
                # Aggressive cleanup every 5 frames (more frequent for high-res)
                if frame_idx % 5 == 0:
                    force_cleanup()
                    # Use print instead of LOGGER for subprocess visibility
                    print(f"Rendered {frame_idx+1}/{num_frames} frames", flush=True)
                
        finally:
            # CRITICAL: Synchronize MPS before closing writers
            # This ensures all GPU operations complete before we release resources
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            elif torch.cuda.is_available():
                torch.cuda.synchronize()
            
            writer_color.close()
            writer_depth.close()
            
            # AGGRESSIVE MPS MEMORY CLEANUP
            # Move gaussians tensors to CPU before deletion to force MPS release
            if gaussians_render is not None:
                try:
                    # Move each tensor field to CPU to release MPS memory
                    for field_name in gaussians_render._fields:
                        field_val = getattr(gaussians_render, field_name)
                        if isinstance(field_val, torch.Tensor) and field_val.device.type in ('mps', 'cuda'):
                            # Create CPU copy and replace
                            setattr(gaussians_render, field_name, None)
                except:
                    pass
                del gaussians_render
                gaussians_render = None
            
            if renderer is not None:
                # Move renderer parameters to CPU
                try:
                    for param in renderer.parameters():
                        param.data = param.data.cpu()
                except:
                    pass
                del renderer
                renderer = None
                
            if camera_model is not None:
                del camera_model
                camera_model = None
            if trajectory is not None:
                # Trajectory is a list of tensors - clear them
                for i in range(len(trajectory)):
                    if isinstance(trajectory[i], torch.Tensor):
                        trajectory[i] = None
                trajectory.clear() if hasattr(trajectory, 'clear') else None
                del trajectory
                trajectory = None
            if intrinsics_batch is not None:
                del intrinsics_batch
                intrinsics_batch = None
            del intrinsics_scaled
            
            # Clear any gsplat internal caches
            if hasattr(gsplat, '_cache'):
                gsplat._cache.clear()
            
            # Force Python to collect garbage first
            gc.collect()
            
            # MPS-specific: create and immediately delete a small tensor
            # This can trigger the allocator to release memory
            if torch.backends.mps.is_available():
                try:
                    torch.mps.synchronize()
                    # Allocate a small tensor to trigger cleanup
                    _dummy = torch.zeros(1, device='mps')
                    del _dummy
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
                except:
                    pass
            
            # Multiple cleanup passes
            for _ in range(3):
                force_cleanup()
            
            # DEBUG: Memory at END of render call
            import psutil
            process = psutil.Process()
            rss_gb = process.memory_info().rss / 1024**3
            traj_name = params.type if params else "default"
            
            mps_info = ""
            if torch.backends.mps.is_available():
                try:
                    mps_alloc = torch.mps.driver_allocated_memory() / 1024**3
                    mps_info = f", MPS Driver: {mps_alloc:.1f}GB"
                except:
                    pass
            
            print(f"=== RENDER END [{traj_name}]: Process RSS={rss_gb:.1f}GB{mps_info} ===", flush=True)

# Patch the original module
sharp_render.render_gaussians = render_gaussians_mps_compatible
render_gaussians = render_gaussians_mps_compatible

# =============================================================================
# OPTIMIZATION: Shared ViT Backbone
# =============================================================================
_vit_cache = {}
_original_create_vit = None

def _install_vit_cache():
    global _original_create_vit
    from sharp.models.encoders import vit_encoder
    from sharp.models.encoders import monodepth_encoder
    
    if _original_create_vit is None:
        _original_create_vit = vit_encoder.create_vit
    
    def cached_create_vit(config=None, preset="dinov2l16_384", intermediate_features_ids=None):
        cache_key = preset
        if cache_key in _vit_cache:
            model = _vit_cache[cache_key]
            if intermediate_features_ids is not None:
                model.intermediate_features_ids = intermediate_features_ids
            LOGGER.info(f"Reusing cached ViT {preset}")
            return model
        
        model = _original_create_vit(config=config, preset=preset, intermediate_features_ids=intermediate_features_ids)
        _vit_cache[cache_key] = model
        LOGGER.info(f"Created and cached ViT {preset}")
        return model
    
    vit_encoder.create_vit = cached_create_vit
    monodepth_encoder.create_vit = cached_create_vit

def _clear_vit_cache():
    global _vit_cache
    _vit_cache.clear()

class MLSharpEngine:
    def __init__(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self.predictor = None
        self.current_checkpoint = None
        self.low_vram = False
        LOGGER.info(f"MLSharpEngine initialized on {self.device}")

    def load_model(self, checkpoint_path=None, low_vram=False):
        if checkpoint_path is None:
            if self.predictor is not None and self.low_vram == low_vram:
                return
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if self.predictor is not None and self.current_checkpoint == checkpoint_path and self.low_vram == low_vram:
            return

        final_checkpoint = checkpoint_path
        if checkpoint_path is not None:
            if low_vram and self.device == "cuda":
                quantized_path = checkpoint_path.with_name(checkpoint_path.stem + "_fp16.pt")
                if quantized_path.exists():
                    final_checkpoint = quantized_path
                else:
                    state_dict = torch.load(checkpoint_path, weights_only=True)
                    for k in state_dict:
                        if isinstance(state_dict[k], torch.Tensor):
                            state_dict[k] = state_dict[k].half()
                    torch.save(state_dict, quantized_path)
                    final_checkpoint = quantized_path

        if final_checkpoint is not None:
            state_dict = torch.load(final_checkpoint, weights_only=True)
        else:
            from sharp.cli.predict import DEFAULT_MODEL_URL
            state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
        
        _clear_vit_cache()
        _install_vit_cache()
        
        self.predictor = create_predictor(PredictorParams())
        self.predictor.load_state_dict(state_dict)
        self.predictor.eval()
        self.predictor.to(self.device)
        
        if low_vram and self.device == "cuda":
             self.predictor.half()
        
        self.current_checkpoint = checkpoint_path
        self.low_vram = low_vram

    def get_image_focal(self, image_path):
        if not image_path: return None
        try:
            from PIL import Image
            img_pil = Image.open(image_path)
            img_exif = io.extract_exif(img_pil)
            f_35mm = img_exif.get("FocalLengthIn35mmFilm", img_exif.get("FocalLenIn35mmFilm", None))
            if f_35mm is None or f_35mm < 1:
                f_35mm = img_exif.get("FocalLength", None)
                if f_35mm is None: return None
                if f_35mm < 10.0: f_35mm *= 8.4
            return float(f_35mm)
        except Exception:
            return None

    @torch.no_grad()
    def predict(self, image_path, output_path, internal_resolution=1536, f_mm_override=None):
        if self.predictor is None:
            raise RuntimeError("Model not loaded.")

        image_path = Path(image_path)
        if f_mm_override is not None: f_35mm = f_mm_override
        else: f_35mm = None

        io_logger = logging.getLogger("sharp.utils.io")
        old_level = io_logger.level
        io_logger.setLevel(logging.ERROR)
        try:
            image, _, f_px_auto = io.load_rgb(image_path)
        finally:
            io_logger.setLevel(old_level)
        
        if f_mm_override is not None:
             f_px = io.convert_focallength(image.shape[1], image.shape[0], f_mm_override)
        else:
             f_px = f_px_auto

        height, width = image.shape[:2]
        internal_shape = (internal_resolution, internal_resolution)
        
        image_pt = torch.from_numpy(image.copy()).float().to(self.device).permute(2, 0, 1) / 255.0
        if self.low_vram and self.device == "cuda":
            image_pt = image_pt.half()
            
        _, h, w = image_pt.shape
        disparity_factor = torch.tensor([f_px / w]).float().to(self.device)
        if self.low_vram and self.device == "cuda":
            disparity_factor = disparity_factor.half()

        image_resized_pt = F.interpolate(
            image_pt[None],
            size=(internal_shape[1], internal_shape[0]),
            mode="bilinear",
            align_corners=True,
        )

        gaussians_ndc = self.predictor(image_resized_pt, disparity_factor)
        dtype = gaussians_ndc.mean_vectors.dtype
        
        # Intrinsics for prediction (NOT for rendering, that's in render_gaussians)
        intrinsics = torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            device=self.device,
            dtype=dtype
        )
        
        intrinsics_resized = intrinsics.clone()
        intrinsics_resized[0] *= internal_shape[0] / width
        intrinsics_resized[1] *= internal_shape[1] / height

        extrinsics = torch.eye(4, device=self.device, dtype=dtype)
        gaussians = unproject_gaussians(
            gaussians_ndc, extrinsics, intrinsics_resized, internal_shape
        )

        save_ply(gaussians, f_px, (height, width), output_path)
        return gaussians, f_px, (width, height)

    def load_gaussians(self, ply_path):
        from sharp.utils.gaussians import load_ply
        return load_ply(Path(ply_path))

    def render_video(self, gaussians, focal_length_px, resolution_px, output_video_path, 
                     trajectory_type="rotate_forward", num_steps=60, render_depth=True, target_short_edge=None,
                     stop_event=None):
        from collections import namedtuple
        
        render_res = resolution_px
        render_f_px = focal_length_px

        # SCALING LOGIC
        if target_short_edge is not None and target_short_edge > 0:
            w, h = resolution_px
            short_edge = min(w, h)
            scale_factor = target_short_edge / short_edge
            
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            new_w = round(new_w / 16) * 16
            new_h = round(new_h / 16) * 16
            
            real_scale_x = new_w / w
            real_scale_y = new_h / h
            real_scale = (real_scale_x + real_scale_y) / 2.0
            
            render_res = (new_w, new_h)
            render_f_px = focal_length_px * real_scale
            LOGGER.info(f"Scaling Render: {resolution_px} -> {render_res} (Mod16), Focal: {focal_length_px:.1f} -> {render_f_px:.1f}")

        if self.device not in ("cuda", "mps"):
            LOGGER.warning(f"Rendering requires CUDA or MPS.")
            return None, None

        lookat_mode = "point"
        if "pan_" in trajectory_type: lookat_mode = "ahead"
            
        # Check cancellation
        if stop_event and stop_event.is_set():
            LOGGER.warning(f"Render cancelled before starting {trajectory_type}")
            return None, None

        params = camera.TrajectoryParams(
            type=trajectory_type,
            num_steps=num_steps,
            lookat_mode=lookat_mode
        )
        
        SceneMetaData = namedtuple('SceneMetaData', ['focal_length_px', 'resolution_px', 'color_space'])
        metadata = SceneMetaData(
            focal_length_px=render_f_px, 
            resolution_px=render_res, 
            color_space="linearRGB"
        )
        
        output_video_path = Path(output_video_path)
        filename_base = output_video_path.stem
        final_video_path = output_video_path.with_name(f"{filename_base}_{trajectory_type}.mp4")
        
        render_gaussians_mps_compatible(
            gaussians=gaussians,
            metadata=metadata,
            params=params,
            output_path=final_video_path,
            stop_event=stop_event
        )
        
        color_video = str(final_video_path)
        depth_video = str(final_video_path.with_suffix(".depth.mp4"))
        if not os.path.exists(depth_video): depth_video = None
        return color_video, depth_video

    def render_video_subprocess(self, ply_path, focal_length_px, resolution_px, output_video_path,
                                 trajectory_type="rotate_forward", num_steps=60, render_depth=True, 
                                 target_short_edge=None, progress_callback=None, stop_event=None):
        """
        Render video using an isolated subprocess.
        
        This approach forces MPS/Metal to release GPU memory when the subprocess exits.
        Each trajectory is rendered in a fresh process, preventing memory accumulation.
        
        Args:
            progress_callback: Optional callable(message: str) for progress updates
        """
        import subprocess
        import json
        
        # Path to the worker script
        worker_script = os.path.join(os.path.dirname(__file__), "render_worker.py")
        
        if not os.path.exists(worker_script):
            LOGGER.warning("render_worker.py not found, falling back to in-process render")
            # Fallback to load_gaussians + render_video
            gaussians, metadata = self.load_gaussians(ply_path)
            return self.render_video(gaussians, focal_length_px, resolution_px, output_video_path,
                                    trajectory_type, num_steps, render_depth, target_short_edge)
        
        output_video_path = Path(output_video_path)
        w, h = resolution_px
        
        # Build command with -u for unbuffered Python output
        cmd = [
            sys.executable,  # Same Python interpreter
            "-u",  # CRITICAL: Unbuffered stdout/stderr for real-time output
            worker_script,
            "--ply_path", str(ply_path),
            "--output_path", str(output_video_path),
            "--trajectory", trajectory_type,
            "--num_steps", str(num_steps),
            "--focal_length_px", str(focal_length_px),
            "--resolution_w", str(w),
            "--resolution_h", str(h),
        ]
        
        if target_short_edge:
            cmd.extend(["--target_short_edge", str(target_short_edge)])
        
        LOGGER.info(f"Spawning subprocess for trajectory: {trajectory_type}")
        
        result_data = None
        
        try:
            # Set unbuffered environment for subprocess
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            
            # Use Popen for real-time output streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                cwd=os.path.dirname(__file__),
                env=env  # Pass unbuffered environment
            )
            
            # Read stdout in real-time
            while True:
                # Check cancellation first
                if stop_event and stop_event.is_set():
                    LOGGER.warning(f"Stop signal received. Terminating subprocess for {trajectory_type}...")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    return None, None

                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line.strip():
                    msg = None  # Message to send to callback
                    
                    # Parse progress from "Rendered X/Y frames"
                    if "Rendered" in line and "/" in line:
                        try:
                            # Extract "Rendered 31/60 frames"
                            parts = line.split("Rendered")[1].split("/")
                            current = int(parts[0].strip())
                            total = int(parts[1].split()[0])
                            pct = (current / total) * 100
                            msg = f"[{trajectory_type}] [Rendering] {current}/{total} frames ({pct:.0f}%)"
                            print(f"[Render {trajectory_type}] {current}/{total} frames ({pct:.0f}%)")
                        except:
                            print(f"[Subprocess] {line.strip()}")
                    elif line.startswith("[Worker] RESULT_JSON:"):
                        json_str = line.replace("[Worker] RESULT_JSON:", "")
                        result_data = json.loads(json_str)
                        msg = f"[{trajectory_type}] [TrajFinished] Render complete!"
                        print(f"[Subprocess] Render complete!")
                    elif "RENDER START" in line:
                        msg = f"[{trajectory_type}] [Rendering] Starting render..."
                        print(f"[Subprocess] {line.strip()}")
                    elif "RENDER END" in line:
                        msg = f"[{trajectory_type}] [TrajFinished] Render finished"
                        print(f"[Subprocess] {line.strip()}")
                    elif "[Worker]" in line:
                        print(f"[Subprocess] {line.strip()}")
                    
                    # Call progress callback if provided
                    if msg and progress_callback:
                        try:
                            progress_callback(msg)
                        except:
                            pass  # Don't crash on callback errors
            
            # Read any remaining stderr
            stderr_output = process.stderr.read()
            if stderr_output:
                for line in stderr_output.split('\n'):
                    if line.strip() and 'Color range not set' not in line:
                        LOGGER.warning(f"[Subprocess stderr] {line}")
            
            # Check result
            if result_data:
                if result_data.get("status") == "success":
                    color_video = result_data.get("color_video")
                    depth_video = result_data.get("depth_video")
                    if depth_video and not os.path.exists(depth_video):
                        depth_video = None
                    return color_video, depth_video
                else:
                    error = result_data.get("error", "Unknown error")
                    LOGGER.error(f"Subprocess render failed: {error}")
                    return None, None
            
            # No result found
            LOGGER.error("No result JSON found in subprocess output")
            return None, None
            
        except subprocess.TimeoutExpired:
            LOGGER.error(f"Subprocess render timed out for {trajectory_type}")
            return None, None
        except Exception as e:
            LOGGER.error(f"Subprocess render error: {e}")
            return None, None

# =============================================================================
# NEW TRAJECTORIES: Dolly and Pan
# Extending ml-sharp capabilities via monkey-patching
# =============================================================================
def create_eye_trajectory_dolly_in(offset_xyz_m, distance_m, num_steps, num_repeats):
    num_steps_total = num_steps * num_repeats
    _, _, offset_z_m = offset_xyz_m
    eye_positions = [torch.tensor([0.0, 0.0, z + distance_m], dtype=torch.float32)
        for z in np.linspace(0.0, offset_z_m, num_steps_total)]
    return eye_positions * num_repeats

def create_eye_trajectory_dolly_out(offset_xyz_m, distance_m, num_steps, num_repeats):
    num_steps_total = num_steps * num_repeats
    _, _, offset_z_m = offset_xyz_m
    eye_positions = [torch.tensor([0.0, 0.0, z + distance_m], dtype=torch.float32)
        for z in np.linspace(offset_z_m, 0.0, num_steps_total)]
    return eye_positions * num_repeats

def create_eye_trajectory_dolly_in_out(offset_xyz_m, distance_m, num_steps, num_repeats):
    num_steps_total = num_steps * num_repeats
    _, _, offset_z_m = offset_xyz_m
    eye_positions = [torch.tensor([0.0, 0.0, distance_m - offset_z_m * np.sin(np.pi * t)], dtype=torch.float32)
        for t in np.linspace(0, num_repeats, num_steps_total)]
    return eye_positions

def create_eye_trajectory_pan_left(offset_xyz_m, distance_m, num_steps, num_repeats):
    num_steps_total = num_steps * num_repeats
    offset_x_m, _, _ = offset_xyz_m
    eye_positions = [torch.tensor([x, 0.0, distance_m], dtype=torch.float32)
        for x in np.linspace(offset_x_m, -offset_x_m, num_steps_total)]
    return eye_positions * num_repeats

def create_eye_trajectory_pan_right(offset_xyz_m, distance_m, num_steps, num_repeats):
    num_steps_total = num_steps * num_repeats
    offset_x_m, _, _ = offset_xyz_m
    eye_positions = [torch.tensor([x, 0.0, distance_m], dtype=torch.float32)
        for x in np.linspace(-offset_x_m, offset_x_m, num_steps_total)]
    return eye_positions * num_repeats

def create_eye_trajectory_pan_left_right(offset_xyz_m, distance_m, num_steps, num_repeats):
    num_steps_total = num_steps * num_repeats
    offset_x_m, _, _ = offset_xyz_m
    eye_positions = [torch.tensor([offset_x_m * np.sin(2 * np.pi * t), 0.0, distance_m], dtype=torch.float32)
        for t in np.linspace(0, num_repeats, num_steps_total)]
    return eye_positions

_original_create_eye_trajectory = camera.create_eye_trajectory
def create_eye_trajectory_extended(scene, params, resolution_px, f_px):
    new_types = {
        "dolly_in": create_eye_trajectory_dolly_in,
        "dolly_out": create_eye_trajectory_dolly_out,
        "dolly_in_out": create_eye_trajectory_dolly_in_out,
        "pan_left": create_eye_trajectory_pan_left,
        "pan_right": create_eye_trajectory_pan_right,
        "pan_left_right": create_eye_trajectory_pan_left_right,
    }
    if params.type in new_types:
        max_offset_xyz_m = camera.compute_max_offset(scene, params, resolution_px, f_px)
        return new_types[params.type](max_offset_xyz_m, params.distance_m, params.num_steps, params.num_repeats)
    else:
        return _original_create_eye_trajectory(scene, params, resolution_px, f_px)

camera.create_eye_trajectory = create_eye_trajectory_extended
