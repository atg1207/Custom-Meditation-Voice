import streamlit as st
import tempfile
import os
import io
import subprocess
from dotenv import load_dotenv
from processing import (load_audiosegment, audiosegment_to_wav_bytes, compute_loudness, match_loudness,
                        apply_serene_processing, replace_segment, SegmentSpec, seconds_from_timestamp, mix_background_audio)
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from openai import OpenAI

# Auto-load .env
load_dotenv()

def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

st.set_page_config(page_title="Meditation Segment Replacer", layout="centered")

st.title("Meditation Command Segment Replacer (OpenAI TTS)")

# --- Check if running on Streamlit Cloud or locally ---
ALLOWED_EXT = (".mp4", ".mp3", ".wav", ".m4a")

# Try to detect local media files, but don't fail if none found
try:
    folder_media = [f for f in os.listdir(os.getcwd()) if f.lower().endswith(ALLOWED_EXT)]
    # Prioritize "The hyno guy.mp4" as default, fallback to alphabetical sort
    if "The hyno guy.mp4" in folder_media:
        default_media_path = "The hyno guy.mp4"
    else:
        folder_media.sort()
        default_media_path = folder_media[0] if folder_media else None
    if default_media_path:
        st.info(f"Auto-detected default media: {default_media_path} (used if you do not upload another file)")
    else:
        st.info("Please upload a media file to get started.")
except:
    # Running on Streamlit Cloud or restricted environment
    default_media_path = None
    st.info("Please upload a media file to get started.")

st.markdown("""
Specify the segment to replace. If your generated or uploaded replacement audio is shorter than the (possibly extended) window, it will be looped seamlessly to fill only that window. The rest of the audio before and after stays untouched. If you request a longer final total duration, only the replacement window is expanded (end point moves later) to achieve the total duration.
""")

with st.expander("TTS Settings", expanded=False):
    voice = st.selectbox("Voice", ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse"], index=1)
    instructions = st.text_area("Voice Instructions (style prompt)", 
                               value="Speak in an extremely soft, gentle, and soothing voice. Use a warm, calm, serene tone with a slow, meditative pace. Sound divine and ethereal, as if speaking from a peaceful dream. Avoid any harshness or sharp sounds.", 
                               height=80)
    
    col1, col2 = st.columns(2)
    with col1:
        divine_mode = st.checkbox("✨ Divine Mode", value=True, help="Adds ethereal echo and reverb for a divine, spiritual quality")
        brightness = st.slider("Serene brightness (post EQ/reverb)", 0.0, 1.0, 0.0, 0.05)
        crossfade_ms = st.slider("Crossfade (ms)", 50, 1000, 300, 10)
        # Background audio controls
        background_audio_enabled = st.checkbox("🎵 Add Background Audio", value=True, help="Mix Background Audio.mp3 with generated TTS")
    with col2:
        loudness_target = st.slider("Fallback Loudness target (LUFS)", -30, -10, -16)
        line_pause = st.slider("Pause between script lines (seconds)", 0.0, 5.0, 0.7, 0.1)
        # Volume controls
        background_volume = st.slider("Background Audio Volume", 0.0, 1.0, 0.7, 0.05, help="Volume of background audio (0=silent, 1=full)")
        custom_audio_volume = st.slider("Custom Audio Volume", 0.0, 1.0, 1.0, 0.05, help="Volume of TTS/custom audio (0=silent, 1=full)")

# Update file upload text based on whether local files are available
upload_label = "Upload video/audio file" if not default_media_path else "(Optional) Upload video/audio to override default"
uploaded_media = st.file_uploader(upload_label, type=["mp4", "mp3", "wav", "m4a"])
col1, col2, col3 = st.columns(3)
start_ts = col1.text_input("Replace Start (HH:MM:SS)", value="00:16:05")
end_ts = col2.text_input("Replace End (HH:MM:SS)", value="00:20:03")
final_len_input = col3.text_input("Target Total Length (HH:MM:SS or blank = keep original)", value="")

mode = st.radio("Replacement Mode", ["Text to TTS", "Upload New Recording"], horizontal=True)

text_input = None
user_audio = None

if mode == "Text to TTS":
    text_input = st.text_area("Replacement Script", height=200, placeholder="Enter your meditation guidance text here... Each new line will have a pause.")
else:
    user_audio = st.file_uploader("Upload replacement audio (wav/mp3)", type=["wav", "mp3", "m4a"])

# Preview and Process buttons
col1, col2 = st.columns(2)
with col1:
    preview_btn = st.button("🎧 Preview", type="secondary", help="Preview the replacement before processing")
with col2:
    submit = st.button("⚡ Process Replacement", type="primary")

client = None
if mode == "Text to TTS":
    try:
        # Try Streamlit secrets first, fallback to environment variables for local development
        try:
            api_key = st.secrets["api_keys"]["OPENAI_API_KEY"]
            if api_key == "your-api-key-here":
                raise KeyError("Placeholder API key found")
        except KeyError:
            # Fallback to environment variables for local development
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("OpenAI API key not found. Please add it to Streamlit Cloud secrets or your .env file.")
                st.stop()
        
        client = OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        pass

# Helper to parse optional final length
def parse_optional_duration(val: str):
    if not val.strip():
        return None
    try:
        return seconds_from_timestamp(val.strip())
    except Exception:
        st.warning("Could not parse target total length; using original length.")
        return None

# Helper function to generate TTS for all lines
def generate_tts_audio(client, lines, voice, instructions, line_pause):
    """Generate TTS audio for all lines with pauses"""
    combined_seg = AudioSegment.silent(duration=0)
    pause_ms = int(line_pause * 1000)
    
    progress = st.progress(0.0)
    for i, line in enumerate(lines, start=1):
        try:
            resp = client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=line,
                instructions=instructions,
            )
            audio_bytes = resp.read() if hasattr(resp, 'read') else resp.content
        except Exception as e:
            st.error(f"TTS request failed on line {i}: {e}")
            st.stop()
        try:
            part_seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        except Exception as e:
            st.error(f"Failed to process TTS audio for line {i}: {e}")
            st.stop()
        combined_seg += part_seg
        if i < len(lines) and pause_ms > 0:
            combined_seg += AudioSegment.silent(duration=pause_ms)
        progress.progress(i / len(lines))
    
    return combined_seg

# Helper function to create preview
def create_preview(base_audio, replacement_audio, start_ms, end_ms, crossfade_ms, preview_duration_ms=10000):
    """Create a short preview of the replacement"""
    # Calculate preview window
    preview_start = max(0, start_ms - 2000)  # 2 seconds before
    preview_end = min(len(base_audio), end_ms + 2000)  # 2 seconds after
    
    # Limit preview duration
    if preview_end - preview_start > preview_duration_ms:
        preview_end = preview_start + preview_duration_ms
    
    # Extract preview segment from base
    preview_base = base_audio[preview_start:preview_end]
    
    # Calculate relative positions in preview
    rel_start = start_ms - preview_start
    rel_end = end_ms - preview_start
    
    # Create preview with replacement
    pre_segment = preview_base[:rel_start]
    post_segment = preview_base[rel_end:]
    
    # Trim or loop replacement to fit
    window_len = rel_end - rel_start
    if len(replacement_audio) < window_len:
        # Loop to fill
        loops_needed = window_len // len(replacement_audio) + 1
        replacement_preview = (replacement_audio * loops_needed)[:window_len]
    else:
        replacement_preview = replacement_audio[:window_len]
    
    # Apply crossfade
    if crossfade_ms > 0 and len(pre_segment) > 0 and len(replacement_preview) > 0:
        cf = min(crossfade_ms, len(pre_segment), len(replacement_preview))
        preview = pre_segment.append(replacement_preview, crossfade=cf)
    else:
        preview = pre_segment + replacement_preview
        
    if len(post_segment) > 0 and crossfade_ms > 0:
        cf = min(crossfade_ms, len(preview), len(post_segment))
        preview = preview.append(post_segment, crossfade=cf)
    else:
        preview = preview + post_segment
    
    return preview

# Handle Preview Button
if preview_btn:
    if not uploaded_media and not default_media_path:
        st.error("No source media available. Upload a file.")
    else:
        try:
            # Load base audio for preview
            if uploaded_media:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_media.name)[1]) as tmp_in:
                    tmp_in.write(uploaded_media.read())
                    tmp_in.flush()
                    base_audio = AudioSegment.from_file(tmp_in.name)
                    os.unlink(tmp_in.name)
            else:
                base_audio = AudioSegment.from_file(default_media_path)
            
            # Parse timestamps
            base_spec = SegmentSpec(start=seconds_from_timestamp(start_ts), end=seconds_from_timestamp(end_ts))
            
            if base_spec.end <= base_spec.start:
                st.error("End must be after start.")
            else:
                # Generate or load replacement audio for preview
                if mode == "Text to TTS":
                    if not text_input or len(text_input.strip()) == 0:
                        st.error("Enter script text for preview.")
                    elif client is None:
                        st.error("OpenAI client not initialized.")
                    else:
                        lines = [ln.strip() for ln in text_input.splitlines() if ln.strip()]
                        if lines:
                            st.info(f"Generating preview with {len(lines)} line(s)...")
                            rep_seg = generate_tts_audio(client, lines, voice, instructions, line_pause)
                            
                            # Apply divine mode processing to preview
                            if divine_mode:
                                rep_wav = audiosegment_to_wav_bytes(rep_seg)
                                rep_wav = apply_serene_processing(rep_wav, brightness=brightness, divine_mode=divine_mode)
                                rep_seg = AudioSegment.from_file(io.BytesIO(rep_wav))
                            
                            # Apply background audio mixing if enabled
                            if background_audio_enabled and os.path.exists("Background Audio.mp3"):
                                rep_seg = mix_background_audio(rep_seg, "Background Audio.mp3", background_volume, custom_audio_volume)
                            
                            # Create preview
                            preview_audio = create_preview(
                                base_audio,
                                rep_seg,
                                int(base_spec.start * 1000),
                                int(base_spec.end * 1000),
                                crossfade_ms,
                                10000  # 10 second preview
                            )
                            
                            # Export preview
                            preview_buf = io.BytesIO()
                            preview_audio.export(preview_buf, format='mp3')
                            preview_buf.seek(0)
                            
                            st.success("Preview ready!")
                            st.audio(preview_buf, format='audio/mp3')
                else:
                    if not user_audio:
                        st.error("Upload replacement audio for preview.")
                    else:
                        rep_seg = AudioSegment.from_file(io.BytesIO(user_audio.read()))
                        
                        # Apply divine mode processing if enabled
                        if divine_mode:
                            rep_wav = audiosegment_to_wav_bytes(rep_seg)
                            rep_wav = apply_serene_processing(rep_wav, brightness=brightness, divine_mode=divine_mode)
                            rep_seg = AudioSegment.from_file(io.BytesIO(rep_wav))
                        
                        # Apply background audio mixing if enabled
                        if background_audio_enabled and os.path.exists("Background Audio.mp3"):
                            rep_seg = mix_background_audio(rep_seg, "Background Audio.mp3", background_volume, custom_audio_volume)
                        
                        # Create preview
                        preview_audio = create_preview(
                            base_audio,
                            rep_seg,
                            int(base_spec.start * 1000),
                            int(base_spec.end * 1000),
                            crossfade_ms,
                            10000  # 10 second preview
                        )
                        
                        preview_buf = io.BytesIO()
                        preview_audio.export(preview_buf, format='mp3')
                        preview_buf.seek(0)
                        
                        st.success("Preview ready!")
                        st.audio(preview_buf, format='audio/mp3')
                        
        except Exception as e:
            st.error(f"Preview error: {str(e)}")

if submit:
    if not uploaded_media and not default_media_path:
        st.error("No source media available. Upload a file.")
    else:
        try:
            base_spec = SegmentSpec(start=seconds_from_timestamp(start_ts), end=seconds_from_timestamp(end_ts))
            if base_spec.end <= base_spec.start:
                st.error("End must be after start.")
            else:
                # Check if ffmpeg is available first
                if not check_ffmpeg():
                    st.error("⚠️ Audio processing tools are not available on this system.")
                    st.error("This may be due to missing system dependencies (ffmpeg).")
                    st.info("Please wait for the system to install required dependencies, or try uploading a WAV file.")
                    st.stop()

                # Load source audio
                try:
                    if uploaded_media:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_media.name)[1]) as tmp_in:
                            tmp_in.write(uploaded_media.read())
                            tmp_in.flush()
                            try:
                                base_audio = AudioSegment.from_file(tmp_in.name)
                                source_name = uploaded_media.name
                                st.success(f"✅ Successfully loaded: {source_name}")
                            except CouldntDecodeError as e:
                                st.error(f"❌ Could not decode the audio file: {str(e)}")
                                st.info("Please try uploading a different audio format (WAV, MP3, M4A recommended)")
                                os.unlink(tmp_in.name)  # Clean up temp file
                                st.stop()
                            except Exception as e:
                                st.error(f"❌ Error loading audio file: {str(e)}")
                                os.unlink(tmp_in.name)  # Clean up temp file
                                st.stop()
                    else:
                        try:
                            base_audio = AudioSegment.from_file(default_media_path)
                            source_name = default_media_path
                            st.success(f"✅ Successfully loaded: {source_name}")
                        except CouldntDecodeError as e:
                            st.error(f"❌ Could not decode the default audio file: {str(e)}")
                            st.stop()
                        except Exception as e:
                            st.error(f"❌ Error loading default audio file: {str(e)}")
                            st.stop()
                except Exception as e:
                    st.error(f"❌ Error handling audio file: {str(e)}")
                    st.stop()
                
                st.write(f"Using source: {source_name}")

                original_length_ms = len(base_audio)
                original_length_sec = original_length_ms / 1000.0
                orig_window_len_sec = base_spec.end - base_spec.start

                original_wav_bytes = audiosegment_to_wav_bytes(base_audio)
                try:
                    orig_lufs = compute_loudness(original_wav_bytes)
                except Exception:
                    orig_lufs = loudness_target
                st.write(f"Original integrated loudness (approx): {orig_lufs:.2f} LUFS")

                # Determine extended spec if needed.
                target_total = parse_optional_duration(final_len_input)
                extend_seconds = 0.0
                if target_total and target_total > original_length_sec:
                    extend_seconds = target_total - original_length_sec
                    st.info(f"Replacement window will be extended by {extend_seconds:.2f}s (only custom audio loops) to reach total {final_len_input}.")
                spec = SegmentSpec(start=base_spec.start, end=base_spec.end + extend_seconds)

                if mode == "Text to TTS":
                    if not text_input or len(text_input.strip()) == 0:
                        st.error("Enter script text.")
                        st.stop()
                    if client is None:
                        st.error("OpenAI client not initialized. Check OPENAI_API_KEY.")
                        st.stop()
                    lines = [ln.strip() for ln in text_input.splitlines() if ln.strip()]
                    if not lines:
                        st.error("Script has no usable lines.")
                        st.stop()
                    st.info(f"Generating {len(lines)} line(s) of TTS with {line_pause:.1f}s pauses…")
                    rep_seg = generate_tts_audio(client, lines, voice, instructions, line_pause)
                else:
                    if not user_audio:
                        st.error("Upload replacement recording.")
                        st.stop()
                    try:
                        rep_seg = AudioSegment.from_file(io.BytesIO(user_audio.read()))
                        st.success("✅ Successfully loaded replacement audio")
                    except CouldntDecodeError as e:
                        st.error(f"❌ Could not decode the replacement audio file: {str(e)}")
                        st.info("Please try uploading a different audio format (WAV, MP3, M4A recommended)")
                        st.stop()
                    except Exception as e:
                        st.error(f"❌ Error loading replacement audio: {str(e)}")
                        st.stop()

                # Process replacement
                try:
                    rep_wav = audiosegment_to_wav_bytes(rep_seg)
                    rep_wav = apply_serene_processing(rep_wav, brightness=brightness, divine_mode=divine_mode)
                    rep_wav = match_loudness(orig_lufs, rep_wav)
                    rep_seg = AudioSegment.from_file(io.BytesIO(rep_wav))
                    
                    # Apply background audio mixing if enabled (only for TTS mode)
                    if mode == "Text to TTS" and background_audio_enabled and os.path.exists("Background Audio.mp3"):
                        rep_seg = mix_background_audio(rep_seg, "Background Audio.mp3", background_volume, custom_audio_volume)
                        
                except Exception as e:
                    st.error(f"❌ Error processing replacement audio: {str(e)}")
                    st.stop()

                # Replace with looping fill
                result_audio = replace_segment(base_audio, rep_seg, spec, crossfade_ms=crossfade_ms)
                if extend_seconds > 0:
                    # Original tail starts at the original base_spec.end timestamp
                    tail_start_ms = int(base_spec.end * 1000)
                    tail = base_audio[tail_start_ms:]
                    result_audio = result_audio + tail
                    # If tail pushes us beyond target, trim
                    target_ms_total = int(target_total * 1000)
                    if len(result_audio) > target_ms_total:
                        result_audio = result_audio[:target_ms_total]

                final_len_sec = len(result_audio) / 1000.0
                st.write(f"Final output length: {final_len_sec/60:.2f} minutes")

                out_buf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                result_audio.export(out_buf.name, format='mp3', bitrate='192k')
                with open(out_buf.name, 'rb') as f:
                    st.success("Replacement complete. Download below.")
                    st.download_button("Download New Full Audio (MP3)", f, file_name="modified_meditation.mp3")
        except Exception as e:
            st.exception(e)

st.markdown("---")
st.caption("Only the replacement window is loop-expanded if a longer total length is requested; pauses inserted between script lines.")