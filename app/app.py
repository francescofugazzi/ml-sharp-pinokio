import gradio as gr
import subprocess
import os
import shutil
import time
import datetime
import glob
import sys
import torch # Only for Cuda check
import json

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
TEMP_DIR = os.path.join(BASE_DIR, "temp_proc")

print(f"DEBUG: BASE_DIR: {BASE_DIR}")
print(f"DEBUG: CONFIG_FILE: {CONFIG_FILE}")

# --- EXEC DETECTION ---
if sys.platform == "win32":
    SHARP_EXE = os.path.join(sys.prefix, "Scripts", "sharp.exe")
else:
    SHARP_EXE = os.path.join(sys.prefix, "bin", "sharp")

if not os.path.exists(SHARP_EXE):
    SHARP_EXE = "sharp"

# Clean init
for d in [OUTPUTS_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

def clean_gradio_cache():
    import tempfile
    # Gradio uses tempfile.gettempdir() / "gradio" by default or GRADIO_TEMP_DIR env var
    gradio_tmp = os.environ.get("GRADIO_TEMP_DIR")
    if not gradio_tmp:
        gradio_tmp = os.path.join(tempfile.gettempdir(), "gradio")
    
    print(f"DEBUG: Cleaning Gradio cache at {gradio_tmp}")
    if os.path.exists(gradio_tmp):
        try:
            # Remove the whole directory and recreate it to ensure it's empty
            shutil.rmtree(gradio_tmp)
            os.makedirs(gradio_tmp, exist_ok=True)
            print("DEBUG: Gradio cache cleared successfully.")
        except Exception as e:
            print(f"DEBUG: Error clearing Gradio cache: {e}")

def clean_temp_dir():
    print(f"DEBUG: Cleaning temp_proc at {TEMP_DIR}")
    if os.path.exists(TEMP_DIR):
        try:
            # Remove files inside but keep the directory
            for f in glob.glob(os.path.join(TEMP_DIR, "*")):
                if os.path.isfile(f) or os.path.islink(f):
                    os.unlink(f)
                elif os.path.isdir(f):
                    shutil.rmtree(f)
            print("DEBUG: temp_proc cleared successfully.")
        except Exception as e:
            print(f"DEBUG: Error clearing temp_proc: {e}")

# Run cleanup on startup
clean_gradio_cache()
clean_temp_dir()

# --- PLY CONVERSION FOR GRADIO ---
import numpy as np

def convert_ply_for_gradio(input_path: str, output_path: str = None) -> str:
    """
    Convert a 3DGS PLY file to Gradio-compatible format.
    Gradio's Gsplat.js only supports float32 properties.
    
    Args:
        input_path: Path to the original PLY file
        output_path: Optional path for the converted file. If None, creates a _gradio.ply file.
    
    Returns:
        Path to the converted PLY file
    """
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        print("DEBUG: plyfile not installed, skipping PLY conversion")
        return input_path
    
    if not os.path.exists(input_path):
        return input_path
        
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_gradio{ext}"
    
    # If converted file already exists and is newer, use it
    if os.path.exists(output_path):
        if os.path.getmtime(output_path) >= os.path.getmtime(input_path):
            print(f"DEBUG: Using cached converted PLY: {output_path}")
            return output_path
    
    try:
        print(f"DEBUG: Converting PLY for Gradio: {input_path}")
        plydata = PlyData.read(input_path)
        vertex = plydata['vertex']
        
        # Get all property names and convert to float32
        props = vertex.data.dtype.names
        num_points = len(vertex.data)
        
        # Create new dtype with all float32
        new_dtype = [(name, 'f4') for name in props]
        new_data = np.empty(num_points, dtype=new_dtype)
        
        # Copy and convert each property
        for name in props:
            new_data[name] = vertex.data[name].astype(np.float32)
        
        # Create new PLY and save
        new_element = PlyElement.describe(new_data, 'vertex')
        PlyData([new_element], text=False).write(output_path)
        
        print(f"DEBUG: PLY converted successfully: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"DEBUG: PLY conversion failed: {e}")
        return input_path

# --- HELPERS ---

def check_cuda():
    avail = torch.cuda.is_available()
    print(f"DEBUG: CUDA Available? {avail}")
    return avail

def load_config():
    print(f"DEBUG: Loading config from {CONFIG_FILE}")
    if not os.path.exists(CONFIG_FILE): 
        print("DEBUG: Config file not found, returning empty.")
        return {}
    try:
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
            print(f"DEBUG: Loaded config data: {data}")
            return data
    except Exception as e:
        print(f"DEBUG: Error loading config: {e}")
        return {}

def save_config(key, value):
    print(f"DEBUG: Saving config {key}={value}")
    cfg = load_config()
    cfg[key] = value
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f)
        print("DEBUG: Config saved successfully.")
    except Exception as e:
        print(f"DEBUG: Error saving config: {e}")

def save_metadata(job_dir, data):
    meta_path = os.path.join(job_dir, "job_info.json")
    try:
        with open(meta_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving metadata: {e}")

def load_metadata(job_dir):
    meta_path = os.path.join(job_dir, "job_info.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except:
            pass
    return {}



def get_history_list():
    """
    Return a list of tuples (thumb_path, caption) for the Gallery.
    """
    items = []
    if not os.path.exists(OUTPUTS_DIR): 
        return []
    
    subdirs = [f.path for f in os.scandir(OUTPUTS_DIR) if f.is_dir()]
    subdirs.sort(key=os.path.getmtime, reverse=True)
    
    for job_dir in subdirs:
        job_name = os.path.basename(job_dir)
        meta = load_metadata(job_dir)
        
        date_str = meta.get("date", "")
        if not date_str:
            ts = os.path.getmtime(job_dir)
            date_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
            
        orig_name = meta.get("original_name", "")
        
        caption = f"{job_name}\nDate: {date_str}\nInput: {orig_name}"
        
        # Find thumbnail
        thumb = None
        candidates = sorted(glob.glob(os.path.join(job_dir, "*input*.*")))
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(job_dir, "*.*")))
            
        for f in candidates:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                thumb = f
                break
        
        if thumb:
            items.append((thumb, caption))
            
    return items

def generate_job_list_html():
    """Generate HTML for job list with clickable rows."""
    items = get_history_list()
    
    if not items:
        return '<div class="job-list-container"><p style="text-align:center;color:gray;">No jobs yet.</p></div>'
    
    html_parts = ['<div class="job-list-container">']
    
    for thumb_path, caption in items:
        lines = caption.split('\n')
        job_name = lines[0] if lines else "Unknown"
        date_str = lines[1].replace("Date: ", "") if len(lines) > 1 else ""
        input_str = lines[2].replace("Input: ", "") if len(lines) > 2 else ""
        
        # Use Gradio's /gradio_api/file= endpoint for local files (like Wan2GP)
        thumb_url = "/gradio_api/file=" + thumb_path.replace("\\", "/")
        
        # Escape single quotes in job name for JavaScript
        safe_job_name = job_name.replace("'", "\\'")
        
        html_parts.append(f'''
        <div class="job-list-item" data-job="{safe_job_name}" onclick="selectJob('{safe_job_name}')">
            <img src="{thumb_url}" alt="thumb" onerror="this.alt='[No Image]'"/>
            <div class="job-info">
                <div class="job-name" title="{job_name}">{job_name}</div>
                <div class="job-meta">{date_str}</div>
                <div class="job-meta" title="{input_str}">Input: {input_str}</div>
            </div>
            <button class="delete-btn" onclick="event.stopPropagation(); deleteJob('{safe_job_name}')" title="Delete Job">✕</button>
        </div>
        ''')
    
    html_parts.append('</div>')
    
    # Note: JavaScript selectJob and deleteJob functions are defined globally in head_js
    
    return ''.join(html_parts)




def get_input_library_items():
    """Dummy function to prevent NameError if still referenced."""
    return []

def predict(image_path, do_render_video):
    if not image_path: return "Error: No image uploaded.", gr.update()
    
    ts = int(time.time())
    original_name = os.path.basename(image_path)
    safe_name, ext = os.path.splitext(original_name)
    
    # 2. Create Job Folder
    job_name = f"{safe_name}_{ts}"
    job_dir = os.path.join(OUTPUTS_DIR, job_name)
    os.makedirs(job_dir, exist_ok=True)
    
    # 3. Save Input in Job (as reference and thumbnail)
    job_input = os.path.join(job_dir, f"input_source{ext}")
    shutil.copy(image_path, job_input)
    
    # 4. Execute Sharp Predict
    # Use temp dir for command input? or direct? Direct is better
    print(f"START JOB: {job_name}")
    
    cmd = [SHARP_EXE, "predict", "-i", job_input, "-o", job_dir]
    
    # Save Initial Metadata
    meta = {
        "job_name": job_name,
        "original_name": original_name,
        "timestamp": ts,
        "date": datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'),
        "status": "processing"
    }
    save_metadata(job_dir, meta)
    
    # Flag --render if requested (and if we have cuda)
    if do_render_video and check_cuda():
        cmd.append("--render")
        # Note: device is set automatically
    elif do_render_video and not check_cuda():
        print("WARNING: Video requested but CUDA not available. Ignoring --render.")

    try:
        subprocess.run(cmd, check=True)
        
        # Check for generated videos
        # Patterns: input_source.mp4 (Color), input_source_depth.mp4 (Depth)
        # Or typical sharp output might vary. Let's look for both.
        
        vid_color = None
        vid_depth = None
        
        # Color
        c_canc = glob.glob(os.path.join(job_dir, "*input_source.mp4"))
        if c_canc: vid_color = c_canc[0]
        
        # Depth
        d_canc = glob.glob(os.path.join(job_dir, "*depth.mp4"))
        if d_canc: vid_depth = d_canc[0]
            
        if d_canc: vid_depth = d_canc[0]
        
        # Update Metadata
        meta["status"] = "completed"
        save_metadata(job_dir, meta)
            
        return f"Completed: {job_name}", gr.update(value=get_history_list()), vid_color, vid_depth, job_dir
    except subprocess.CalledProcessError as e:
        return f"CLI Error: {e}", gr.update(value=get_history_list()), None, None, None
    except Exception as e:
        return f"Generic Error: {e}", gr.update(value=get_history_list()), None, None, None

def render_video_for_job(job_name):
    if not job_name: return "No job selected", None, None
    
    job_dir = os.path.join(OUTPUTS_DIR, job_name)
    # Find PLY
    plys = glob.glob(os.path.join(job_dir, "*.ply"))
    if not plys: return "No PLY file found in this job.", None
    
    ply_path = plys[0]
    
    if not check_cuda():
        return "Error: Video rendering requires CUDA (NVIDIA GPU).", None, None
        
    cmd = [SHARP_EXE, "render", "-i", ply_path, "-o", job_dir]
    try:
        subprocess.run(cmd, check=True)
        # Find generated video
        vid_color = None
        vid_depth = None
        
        # Color
        c_canc = glob.glob(os.path.join(job_dir, "*input_source.mp4"))
        if c_canc: vid_color = c_canc[0]
        
        # Depth
        d_canc = glob.glob(os.path.join(job_dir, "*depth.mp4"))
        if d_canc: vid_depth = d_canc[0]
            
        return "Video(s) Rendered!", vid_color, vid_depth
        
    except Exception as e:
        return f"Render Error: {e}", None, None

def load_job_details(evt: gr.SelectData):
    # evt.value is the caption if gallery mode is caption. But here we pass tuples (img, label)
    # If we use tuples in gallery, evt.value is the Label (job_name) if defined?
    # Gradio Gallery returns index mostly.
    
    # Retrieve job name from current list
    # In simpler mode: use index
    # For Dataframe: evt.index is [row_index, col_index]
    if isinstance(evt.index, (list, tuple)):
        row_idx = evt.index[0]
    else:
        row_idx = evt.index
        
    row_idx = evt.index
        
    all_rows = get_history_list()
    if row_idx >= len(all_rows): return None, None, [], None, None, None
    
    # item is (thumb, caption)
    # caption is "JobName\n..."
    caption = all_rows[row_idx][1]
    selected_job_name = caption.split('\n')[0] 
    job_dir = os.path.join(OUTPUTS_DIR, selected_job_name)
    
    # 1. Input Img
    input_img = None
    
    # 2. Files
    files = glob.glob(os.path.join(job_dir, "*.*"))
    
    candidates = sorted(glob.glob(os.path.join(job_dir, "*input*.*")))
    if not candidates: candidates = sorted(glob.glob(os.path.join(job_dir, "*.*")))
    
    for f in candidates:
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            input_img = f
            break
            
    vid_color = None
    vid_depth = None
    file_list = []
    ply_file = None
    
    for f in files:
        file_list.append(f)
        if f.endswith(".ply"): ply_file = f
        if f.endswith("input_source.mp4"): vid_color = f
        if f.endswith("depth.mp4"): vid_depth = f
        
    # Status Video Button
    vid_status_msg = ""
    if vid_color or vid_depth:
        vid_status_msg = "Video available."
        vid_btn_interactive = True # Può voler rigenerare
    else:
        vid_status_msg = "No video present."
        vid_btn_interactive = True
        

    return (
        selected_job_name,        # Label
        input_img,                # Image Preview
        file_list,                # File links
        vid_color,                # Video Color
        vid_depth,                # Video Depth
        f"Status: {vid_status_msg}" # Log
    )

# --- JOB HELPERS (Global Scope) ---

# Global state for task tracking
running_tasks = {}

def is_system_busy():
    return len(running_tasks) > 0

def get_busy_message():
    if not running_tasks: return ""
    # Return first active task status
    job, status = next(iter(running_tasks.items()))
    return f"System Busy: Processing '{job}' ({status})"

# --- JOB HELPERS (Global Scope) ---
def zip_job(job_name):
    # helper for error return (refresh but keep zip hidden)
    def return_error():
         return load_job_details_by_name(job_name)

    if not job_name: 
        yield return_error()
        return
        
    if is_system_busy():
        print(f"DEBUG: System busy, rejecting zip_job for {job_name}")
        yield return_error()
        return

    job_dir = os.path.join(OUTPUTS_DIR, job_name)
    if not os.path.exists(job_dir): 
        yield return_error()
        return
    
    # Set running state
    running_tasks[job_name] = "Creating ZIP archive..."
    
    try:
        # Yield 1: Processing
        # [lbl, img, model3d, files, vc, vd, log, zip_out, ply_btn, vid_btn, make_zip_btn, del_btn, run_btn]
        yield (
            gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            running_tasks[job_name], 
            gr.update(visible=False), 
            gr.update(interactive=False), gr.update(interactive=False), # Lock other actions
            gr.update(interactive=False), # Lock self (zip)
            gr.update(interactive=False), # Lock delete
            gr.update(interactive=False)  # Lock global run
        )
        
        # Create zip in TEMP_DIR
        zip_base = os.path.join(TEMP_DIR, job_name)
        archive_path = shutil.make_archive(zip_base, 'zip', job_dir)
        full_path = archive_path
        
        # Clear state
        if job_name in running_tasks: del running_tasks[job_name]
        
        # Get standard refresh state
        fresh_state = list(load_job_details_by_name(job_name))
        
        # Override zip output (index 7 now)
        fresh_state[7] = gr.update(value=full_path, visible=True, label=f"Download {os.path.basename(full_path)}")
        fresh_state[6] = "ZIP Created Successfully"
        
        yield tuple(fresh_state)
        
    except Exception as e:
        print(f"DEBUG: Error creating zip: {e}")
        if job_name in running_tasks: del running_tasks[job_name]
        yield return_error()

def retry_job_ply(job_name):
    # helper for return
    def return_refresh(log_msg=""):
        res = list(load_job_details_by_name(job_name))
        if log_msg: res[5] = log_msg
        return tuple(res)
        
    if not job_name: 
        yield return_refresh("No job selected")
        return
        
    if is_system_busy():
        yield return_refresh(get_busy_message())
        return

    job_dir = os.path.join(OUTPUTS_DIR, job_name)
    
    # ... (input finding logic same) ...
    input_file = None
    candidates = glob.glob(os.path.join(job_dir, "*"))
    for f in candidates:
        if "input" in os.path.basename(f).lower() and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            input_file = f
            break
            
    if not input_file:
        for f in candidates:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                input_file = f
                break
    
    if not input_file:
        yield return_refresh("Error: No valid input image found.")
        return
        
    print(f"RETRY PLY JOB: {job_name} using {input_file}")
    
    # Set state
    running_tasks[job_name] = "Generating PLY..."
    
    try:
        # Yield 1: Processing
        yield (
            gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            running_tasks[job_name] + " (please wait)", 
            gr.skip(), 
            gr.update(interactive=False), # self (ply)
            gr.update(interactive=False), # vid
            gr.update(interactive=False), # zip
            gr.update(interactive=False), # delete
            gr.update(interactive=False)  # run
        )
        
        cmd = [SHARP_EXE, "predict", "-i", input_file, "-o", job_dir]
        subprocess.run(cmd, check=True)
        
        if job_name in running_tasks: del running_tasks[job_name]
        yield return_refresh(f"PLY Generated for {job_name}")
        
    except Exception as e:
        if job_name in running_tasks: del running_tasks[job_name]
        yield return_refresh(f"Error generating PLY: {e}")

def regen_video_action(job_name):
    if is_system_busy():
        res = list(load_job_details_by_name(job_name))
        res[6] = get_busy_message()  # Index 6 is log now
        yield tuple(res)
        return

    running_tasks[job_name] = "Rendering Video..."
    try:
        # Yield 1: Processing
        yield (
            gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            running_tasks[job_name] + " (This is heavy)", 
            gr.skip(), 
            gr.skip(), # ply
            gr.update(interactive=False), # self (video)
            gr.update(interactive=False), # zip
            gr.update(interactive=False), # delete
            gr.update(interactive=False)  # run
        )

        msg, vc, vd = render_video_for_job(job_name)
        
        if job_name in running_tasks: del running_tasks[job_name]
        
        res = list(load_job_details_by_name(job_name))
        res[6] = msg  # Index 6 is log
        yield tuple(res)
    except Exception as e:
        if job_name in running_tasks: del running_tasks[job_name]
        res = list(load_job_details_by_name(job_name))
        res[6] = f"Error: {e}"  # Index 6 is log
        yield tuple(res)

def load_job_details_by_name(job_name):
    """Load job details by job name."""
    
    # GLOBAL LOCK CHECK for Buttons
    busy_msg = ""
    lock_ui = False
    
    if is_system_busy():
        lock_ui = True
        if job_name in running_tasks:
            busy_msg = f"⚠️ BUSY: {running_tasks[job_name]}"
        else:
            busy_msg = f"⚠️ SYSTEM BUSY: {get_busy_message()}"
    
    # Defaults
    zip_btn_upd = gr.update(interactive=True)
    run_btn_upd = gr.update(interactive=True)

    if not job_name:
         # lbl, img, model3d, files, vc, vd, log, zip_file, ply_btn, vid_btn, zip_btn, del_btn, run_btn
        return "**No job selected**", None, None, None, None, None, "Select a job.", gr.update(visible=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), run_btn_upd
    
    job_dir = os.path.join(OUTPUTS_DIR, job_name)
    if not os.path.exists(job_dir):
        return f"**{job_name}** (not found)", None, None, None, None, None, "Job folder not found.", gr.update(visible=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), run_btn_upd
    
    # Find input image
    input_img = None
    candidates = glob.glob(os.path.join(job_dir, "*"))
    for f in candidates:
         if "input" in os.path.basename(f).lower() and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            input_img = f
            break
    if not input_img:
        for f in candidates:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                input_img = f
                break
    
    all_files = sorted(candidates)
    has_ply = any(f.endswith(".ply") for f in all_files)
    
    # Find PLY file for 3D viewer
    ply_file = None
    for f in all_files:
        if f.lower().endswith(".ply") and "_gradio" not in f.lower():
            ply_file = f
            break
    
    # Convert PLY to Gradio-compatible format
    if ply_file:
        ply_file = convert_ply_for_gradio(ply_file)
        print(f"DEBUG: PLY file for viewer {job_name}: {ply_file}")
    
    vid_color = None
    vid_depth = None
    for f in all_files:
        fname = os.path.basename(f).lower()
        if fname == "input_video.mp4" or fname.endswith("color.mp4"): vid_color = f
        elif fname.endswith("depth.mp4"): vid_depth = f
    
    if not vid_color:
        for f in all_files:
            if f.lower().endswith('.mp4') and 'depth' not in f.lower():
                vid_color = f
                break
    
    vid_status = "Video available." if (vid_color or vid_depth) else "No video present."
    
    # Defaults
    zip_file_update = gr.update(value=None, visible=False)
    
    # Lock logic overrides
    if lock_ui:
        ply_update = gr.update(interactive=False)
        vid_update = gr.update(interactive=False)
        zip_btn_upd = gr.update(interactive=False)
        del_btn_upd = gr.update(interactive=False)
        run_btn_upd = gr.update(interactive=False)
        status_text = busy_msg
    else:
        ply_update = gr.update(interactive=True)
        can_render = has_ply and check_cuda()
        vid_update = gr.update(interactive=can_render)
        zip_btn_upd = gr.update(interactive=True)
        del_btn_upd = gr.update(interactive=True)
        run_btn_upd = gr.update(interactive=True)
        status_text = f"Status: {vid_status}"
    
    return (
        f"**{job_name}**",
        input_img,
        ply_file,  # For det_model_3d
        all_files,
        vid_color,
        vid_depth,
        status_text,
        zip_file_update,
        ply_update,
        vid_update,
        zip_btn_upd, # Btn Make Zip
        del_btn_upd, # Btn Delete (Right)
        run_btn_upd  # Btn Run (New Job)
    )

# ...

# Inside EVENTS ... 
    # 4. Job Execution - also refresh the HTML list
    def predict_and_refresh(image_path, do_render_video):
        if is_system_busy():
            return get_busy_message(), gr.update(), None, None
        
        result = predict(image_path, do_render_video)
        log_msg = result[0]
        vid_c = result[2] if len(result) > 2 else None
        vid_d = result[3] if len(result) > 3 else None
        job_dir = result[4] if len(result) > 4 else None
        
        # Gather Result Files (PLY and MP4s) for Tab 1
        files_found = []
        if job_dir and os.path.exists(job_dir):
             raw_files = glob.glob(os.path.join(job_dir, "*.*"))
             for f in raw_files:
                 if f.lower().endswith(('.ply', '.mp4')):
                     files_found.append(f)
        else:
             pass 
        
        # Fallback if no job_dir but we have video (rare)
        if not files_found and vid_c:
             job_dir_path = os.path.dirname(vid_c)
             raw_files = glob.glob(os.path.join(job_dir_path, "*.*"))
             for f in raw_files:
                 if f.lower().endswith(('.ply', '.mp4')):
                     files_found.append(f)

        
        new_html = generate_job_list_html()
        return log_msg, new_html, vid_c, vid_d, files_found


# --- UI ---
theme = gr.themes.Ocean()
# --- ASSETS LOADING ---
def load_assets():
    css_path = os.path.join(BASE_DIR, "style.css")
    js_path = os.path.join(BASE_DIR, "script.js")
    
    css_content = ""
    js_content = ""
    
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
            
    if os.path.exists(js_path):
        with open(js_path, "r", encoding="utf-8") as f:
            raw_js = f.read()
            js_content = f"<script>{raw_js}</script>"
            
    return css_content, js_content

app_css, app_js = load_assets()

with gr.Blocks(title="WebUI for ML-Sharp (3DGS)", delete_cache=(86400, 86400)) as demo:
    gr.Markdown("# WebUI for ML-Sharp (3DGS)")
    gr.HTML("<h3 style='text-align: center;'>Implementation of SHARP: Sharp Monocular View Synthesis in Less Than a Second</h3>")
    
    with gr.Tabs(elem_id="main_tabs") as main_tabs:
        
        # --- TAB 1: NEW JOB ---
        with gr.Tab("New Job"):
            with gr.Row():
                with gr.Column(scale=1):
                    new_input = gr.Image(label="Upload Image", type="filepath")
                    has_cuda = check_cuda()
                    # Load default preference, but force False if no CUDA
                    default_render = load_config().get("render_video", False) and has_cuda
                    
                    chk_render = gr.Checkbox(label="Generate Video Immediately (Requires CUDA)", value=default_render, interactive=has_cuda)
                    if not has_cuda:
                        gr.Markdown("*CUDA not detected: Video rendering disabled.*")
                    
                    # Save preference on change
                    def on_render_change(val):
                        save_config("render_video", val)
                    
                    chk_render.change(fn=on_render_change, inputs=[chk_render], outputs=[])
                    
                    
                    btn_run = gr.Button("Start Generation", variant="primary")
                    new_log = gr.Textbox(label="Execution Log")

                with gr.Column(scale=2):
                    # 3D Model Viewer - Camera settings matching Apple ML-Sharp rotate_forward start
                    new_model_3d = gr.Model3D(
                        label="3D Gaussian Splat Preview",
                        height=500,
                        camera_position=(0, 0, 1),  # Frontal view (alpha=0°, beta=0°, radius=2.5)
                        zoom_speed=0.5,
                        pan_speed=0.5,
                        interactive=False
                    )
                    
                    # Result Files (PLY + MP4) - Download Only
                    new_result_files = gr.File(label="Generated Files (3DGS PLY & Video)", file_count="multiple", interactive=False, elem_id="new_result_files_list")
                    
                    with gr.Row():
                        new_vid_c = gr.Video(label="Video", interactive=False, loop=True, autoplay=True)
                        new_vid_d = gr.Video(label="Depth Video", interactive=False, loop=True, autoplay=True)

        # --- TAB 2: HISTORY ---
        with gr.Tab("Result History"):
            with gr.Row():
                # LEFT COLUMN: Job List (Custom HTML)
                with gr.Column(scale=1):
                    gr.Markdown("### Recent Jobs")
                    # Custom HTML list with clickable rows
                    hist_list_html = gr.HTML(value=generate_job_list_html())
                    # Textbox for receiving selection and delete from JavaScript
                    # Use visible=True with CSS hiding so they exist in DOM
                    with gr.Row(visible=True, elem_classes="hidden-controls"):
                        job_selector = gr.Textbox(label="", elem_id="job_selector_input", container=False)
                        job_delete = gr.Textbox(label="", elem_id="job_delete_input", container=False)
                        file_delete = gr.Textbox(label="", elem_id="file_delete_input", container=False)
                
                # RIGHT COLUMN: Selected Job Details
                with gr.Column(scale=2):
                    gr.Markdown("### Job Details")
                    selected_job_lbl = gr.Markdown("**No job selected**")
                    
                    with gr.Row():
                        det_img = gr.Image(label="Original Input", interactive=False, height=400)
                    
                    # 3D Model Viewer for History - matching Apple ML-Sharp rotate_forward start
                    det_model_3d = gr.Model3D(
                        label="3D Gaussian Splat",
                        height=400,
                        camera_position=(0, 0, 2.5),  # Frontal view (alpha=0°, beta=0°, radius=2.5)
                        zoom_speed=0.5,
                        pan_speed=0.5,
                        interactive=False
                    )
                    
                    with gr.Row():
                        det_vid_c = gr.Video(label="Video", interactive=False, height=400, loop=True, autoplay=True)
                        det_vid_d = gr.Video(label="Depth Video", interactive=False, height=400, loop=True, autoplay=True)
                    
                    # New Action Buttons Row
                    with gr.Row():
                        btn_make_zip = gr.Button("📦 Create ZIP Archive", variant="secondary", interactive=False)
                        btn_gen_ply = gr.Button("⚙️ Generate 3DGS PLY", variant="primary", interactive=False)
                        btn_gen_video = gr.Button("🎥 Generate Video", variant="secondary", interactive=False)
                    
                    # Zip Output File (Hidden initially)
                    zip_output = gr.File(label="Download ZIP", interactive=False, visible=False)
                        
                    det_files = gr.File(label="All Files (Download)", file_count="multiple", interactive=False, elem_id="det_files_list")
                    
                    with gr.Row():
                        btn_delete_job = gr.Button("🗑️ Delete Job", variant="stop", elem_id="btn_delete_job_details", interactive=False)
                    
                    det_log = gr.Textbox(label="Operation Status", lines=1)

        # --- TAB 3: LICENSES & CREDITS ---
        with gr.Tab("Licenses & Credits"):
            gr.Markdown("### Credits & Documentation")
            
            def get_repo_file_content(filename):
                # The repo is cloned into app/ml-sharp, which is BASE_DIR/ml-sharp
                target_path = os.path.join(BASE_DIR, "ml-sharp", filename)
                if os.path.exists(target_path):
                    try:
                        with open(target_path, "r", encoding="utf-8") as f:
                            return f.read()
                    except Exception as e:
                        return f"Error reading {filename}: {e}"
                return f"File '{filename}' not found. It should be available after the full installation process."

            with gr.Accordion("ML-Sharp Guide (README)", open=True):
                gr.Markdown(get_repo_file_content("README.md"))

            with gr.Accordion("Software License (LICENSE)", open=False):
                gr.Code(value=get_repo_file_content("LICENSE"), language="markdown", interactive=False)
            
            with gr.Accordion("Model License (LICENSE_MODEL)", open=False):
                gr.Code(value=get_repo_file_content("LICENSE_MODEL"), language="markdown", interactive=False)
                
            with gr.Accordion("Acknowledgements & Credits", open=False):
                gr.Markdown(get_repo_file_content("ACKNOWLEDGEMENTS"))

        # --- TAB 4: APP GUIDE ---
        with gr.Tab("App Guide"):
            def get_project_readme():
                target_path = os.path.join(BASE_DIR, "..", "README.md")
                if os.path.exists(target_path):
                    try:
                        with open(target_path, "r", encoding="utf-8") as f:
                            return f.read()
                    except Exception as e:
                        return f"Error reading README.md: {e}"
                return "README.md not found in the project root."
            
            gr.Markdown(get_project_readme())

    
    # State for current job
    current_job_state = gr.State("")
    
    # State for library selection (Removed)
    # selected_lib_item = gr.State(None)
    
    # EVENTS
    
    # 3. Job Selection

    # The output list must match load_job_details_by_name returns: 
    # Common output list for all refresh actions (now Includes ZIP and RUN buttons)
    refresh_outputs = [
        selected_job_lbl, 
        det_img, 
        det_model_3d,  # NEW: 3D Viewer
        det_files, 
        det_vid_c, 
        det_vid_d, 
        det_log, 
        zip_output,     
        btn_gen_ply, 
        btn_gen_video,
        btn_make_zip,   # Index 10
        btn_delete_job, # Index 11
        btn_run         # Index 12
    ]

    # 3. Job Selection
    # The output list must match load_job_details_by_name returns (11 items) + state = 12
    def on_job_selected(job_name):
        print(f"[DEBUG] on_job_selected called with: '{job_name}'")
        if not job_name:
             # Match return count: 12 + 1 = 13 (refresh_outputs has 12 items)
             # Use load_job_details_by_name("") to get correct defaults
             return load_job_details_by_name("") + ("",)
        result = load_job_details_by_name(job_name)
        return result + (job_name,)
        
    job_selector.change(
        fn=on_job_selected,
        inputs=[job_selector],
        outputs=refresh_outputs + [current_job_state] # 12 + 1 = 13
    )
    
    # 4. Job Execution - GENERATOR with Global Lock
    def predict_and_refresh(image_path, do_render_video, current_job_name):
        if is_system_busy():
            # Error yield: Update log, keep everything else same
            yield ("⚠️ SYSTEM BUSY", gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())
            return
            
        # Set Lock
        new_job_task = "Training New Job..."
        task_key = "NEW_JOB_PENDING"
        running_tasks[task_key] = new_job_task
        
        try:
             # Yield 1: BUSY STATE
             # Outputs: [new_log, new_model_3d, hist_list_html, new_vid_c, new_vid_d, new_result_files, btn_run, btn_make_zip, btn_gen_ply, btn_gen_video]
             yield (
                 "Starting training... (System Locked)", 
                 gr.skip(),  # new_model_3d
                 gr.skip(), gr.skip(), gr.skip(), gr.skip(), # html, vids, files 
                 gr.update(interactive=False), # btn_run
                 gr.update(interactive=False), # btn_make_zip
                 gr.update(interactive=False), # btn_gen_ply
                 gr.update(interactive=False)  # btn_gen_video
             )
             
             result = predict(image_path, do_render_video)
             log_msg = result[0]
             vid_c = result[2] if len(result) > 2 else None
             vid_d = result[3] if len(result) > 3 else None
             job_dir = result[4] if len(result) > 4 else None
             
             # Find PLY for 3D viewer
             ply_file = None
             files_found = []
             if job_dir and os.path.exists(job_dir):
                  raw_files = glob.glob(os.path.join(job_dir, "*.*"))
                  for f in raw_files:
                      # Skip converted files for file list
                      if "_gradio" in f.lower():
                          continue
                      if f.lower().endswith('.ply'):
                          ply_file = f
                      if f.lower().endswith(('.ply', '.mp4')):
                          files_found.append(f)
             
             # Convert PLY for Gradio viewer
             if ply_file:
                  ply_file = convert_ply_for_gradio(ply_file)
             
             new_html = generate_job_list_html()
             
             # Unset lock
             if task_key in running_tasks: del running_tasks[task_key]
             
             # Restore Right Panel State
             right_state = load_job_details_by_name(current_job_name)
             
             # Indices updated: 0=lbl, 1=img, 2=model3d, 3=files, 4=vc, 5=vd, 6=log, 7=zip, 8=ply_btn, 9=vid_btn, 10=zip_btn, 11=del_btn, 12=run_btn
             ply_upd = right_state[8]
             vid_upd = right_state[9]
             zip_upd = right_state[10]
             del_upd = right_state[11]
             run_upd = right_state[12]
             
             # Yield 2: Completed State
             # Outputs: [new_log, new_model_3d, hist_list_html, new_vid_c, new_vid_d, new_result_files, btn_run, btn_make_zip, btn_gen_ply, btn_gen_video]
             yield (
                 log_msg, 
                 ply_file,        # new_model_3d
                 new_html, 
                 vid_c, 
                 vid_d, 
                 files_found,     # new_result_files
                 run_upd,         # btn_run
                 zip_upd,         # btn_make_zip
                 ply_upd,         # btn_gen_ply
                 vid_upd          # btn_gen_video
             )
             
        except Exception as e:
            if task_key in running_tasks: del running_tasks[task_key]
            yield (f"Error: {e}", gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True))

    # 4. Buttons
    btn_run.click(
        predict_and_refresh, 
        inputs=[new_input, chk_render, current_job_state], 
        outputs=[
             new_log, 
             new_model_3d, # NEW: 3D Viewer
             hist_list_html, 
             new_vid_c, 
             new_vid_d,
             new_result_files, 
             btn_run,      # Self
             btn_make_zip, # Right side
             btn_gen_ply,  # Right side
             btn_gen_video # Right side
        ]
    )
    
    # ZIP Generation
    btn_make_zip.click(
        fn=zip_job,
        inputs=[current_job_state],
        outputs=refresh_outputs
    )
    
    btn_gen_ply.click(
        fn=retry_job_ply,
        inputs=[current_job_state],
        outputs=refresh_outputs
    )
    
    btn_gen_video.click(
        fn=regen_video_action,
        inputs=[current_job_state],
        outputs=refresh_outputs
    )
    
    # 6. Delete functionality
    def delete_job_action(job_name):
        if not job_name:
             # Return updates preserving current UI but clearing state if needed
             return (gr.update(),) * 13 + ("",)
            
        job_dir = os.path.join(OUTPUTS_DIR, job_name)
        if os.path.exists(job_dir):
            try:
                shutil.rmtree(job_dir)
                msg = f"Deleted {job_name}"
            except Exception as e:
                msg = f"Error deleting: {e}"
        else:
            msg = "Job not found"
        
        # Refresh HTML list and clear details
        new_html = generate_job_list_html()
        
        # Get default "empty" state for right panel
        default_state = list(load_job_details_by_name(""))
        # Index 5 is status_text/log
        default_state[5] = msg
        
        # Return new_html + default_state (12 items) + empty state string
        # Match order: hist_list_html (1) + load_job_details_by_name (12) + current_job_state (1) = 14 items
        return (new_html,) + tuple(default_state) + ("",)

    btn_delete_job.click(
        fn=None,
        inputs=[],
        outputs=[],
        js="triggerDeleteCurrentJob"  # Trigger delete via JS modal
    )
    
    # Also wire up inline delete from X button (triggered via job_delete textbox)
    # Also wire up inline delete from X button (triggered via job_delete textbox)
    job_delete.change(
        delete_job_action,
        inputs=[job_delete],
        outputs=[hist_list_html] + refresh_outputs + [current_job_state] # 1 + 12 + 1 = 14
    )
    
    # 7. Single File Deletion
    def delete_single_file(job_name, file_name_raw):
        print(f"DEBUG: delete_single_file called with job='{job_name}'")
        
        # helper for return
        def return_refresh(log_msg=""):
            res = list(load_job_details_by_name(job_name))
            if log_msg: res[5] = log_msg
            return tuple(res)
            
        if not job_name or not file_name_raw:
            return return_refresh("Error: No file specified")
        
        if is_system_busy():
            return return_refresh(get_busy_message())
            
        file_name_clean = file_name_raw.replace('\n', '').replace('\r', '').strip()
        base_name = os.path.basename(file_name_clean)
        
        job_dir = os.path.join(OUTPUTS_DIR, job_name)
        file_path = os.path.join(job_dir, base_name)
        
        # Fuzzy match (simplified for this update)
        if not os.path.exists(file_path):
             valid_files = glob.glob(os.path.join(job_dir, "*"))
             for f in sorted(valid_files, key=len, reverse=True):
                 if base_name.startswith(os.path.basename(f)):
                     file_path = f
                     break

        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                msg = f"Deleted file: {base_name}"
            else:
                msg = f"File not found: {base_name}"
        except Exception as e:
            msg = f"Error deleting file: {e}"
            
        return return_refresh(msg)

    file_delete.change(
        delete_single_file,
        inputs=[job_selector, file_delete],
        outputs=refresh_outputs
    )
    
    # 6. Input Library Logic
    
    # Select from Library -> Update State Only (User must click "Use" to confirm)
    # Library Logic Removed 


    # Initial Load - Triggered at the end
    # Use a Timer to trigger load to avoid race conditions
    timer = gr.Timer(value=1.0, active=True)
    
    def initial_load():
        html = generate_job_list_html()
        # libs = get_input_library_items()
        return gr.update(value=html), gr.Timer(active=False)
        
    timer.tick(initial_load, outputs=[hist_list_html, timer])

if __name__ == "__main__":
    native_allowed = [BASE_DIR, OUTPUTS_DIR]
    demo.launch(server_name="127.0.0.1", allowed_paths=native_allowed, theme=theme, css=app_css, head=app_js)