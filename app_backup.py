import streamlit as st
import tempfile
import os
import io
import subprocess
from dotenv import load_dotenv
from processing import (load_audiosegment, audiosegment_to_wav_bytes, compute_loudness, match_loudness,
                        apply_serene_processing, replace_segment, SegmentSpec, seconds_from_timestamp)
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from openai import OpenAI
from audio_analysis import (
    extract_segment_audio, analyze_voice_characteristics, 
    match_voice_profile, extract_ambience, blend_with_ambience,
    apply_de_esser
)
from preview_mixer import (
    PreviewMixer, MusicLibrary, create_preview_ui, 
    create_music_mixer_ui
)
import numpy as np
import soundfile as sf

# Auto-load .env
load_dotenv()

# Initialize preview mixer and music library
if 'preview_mixer' not in st.session_state:
    st.session_state.preview_mixer = PreviewMixer()
if 'music_library' not in st.session_state:
    st.session_state.music_library = MusicLibrary()

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
                               value="Speak in an extremely soft, gentle, and soothing whisper. Use a warm, calm, serene tone with a slow, meditative pace. Sound divine and ethereal, as if speaking from a peaceful dream. Avoid any harshness or sharp sounds.", 
                               height=80)
    
    col1, col2 = st.columns(2)
    with col1:
        divine_mode = st.checkbox("✨ Divine Mode", value=True, help="Adds ethereal echo and reverb for a divine, spiritual quality")
        brightness = st.slider("Serene brightness (post EQ/reverb)", 0.0, 1.0, 0.6, 0.05)
        crossfade_ms = st.slider("Crossfade (ms)", 50, 1000, 300, 10)
    with col2:
        auto_analyze = st.checkbox("🎯 Auto-Analyze Original", value=True, help="Automatically analyze and match the original segment's characteristics")
        loudness_target = st.slider("Fallback Loudness target (LUFS)", -30, -10, -16)
        line_pause = st.slider("Pause between script lines (seconds)", 0.0, 5.0, 0.7, 0.1)

# Voice Matching Settings
with st.expander("Voice Matching (Advanced)", expanded=False):
    enable_voice_matching = st.checkbox("Enable Voice Profile Matching", value=False, 
                                       help="Analyze and match the voice characteristics of the original segment")
    if enable_voice_matching:
        col1, col2 = st.columns(2)
        with col1:
            match_pitch = st.checkbox("Match Pitch", value=True)
            match_tempo = st.checkbox("Match Speaking Rate", value=True)
            match_timbre = st.checkbox("Match Timbre/Brightness", value=True)
        with col2:
            match_dynamics = st.checkbox("Match Dynamic Range", value=True)
            match_reverb = st.checkbox("Match Room Acoustics", value=False)
            preserve_ambience = st.checkbox("Preserve Background Ambience", value=True)

# Preview Settings
preview_settings = create_preview_ui(st.session_state.preview_mixer)

# Music Mixing Settings  
music_settings = create_music_mixer_ui(st.session_state.music_library)

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
col1, col2, col3 = st.columns(3)
with col1:
    preview_btn = st.button("🎧 Preview", type="secondary", help="Preview the replacement before processing")
with col2:
    submit = st.button("⚡ Process Replacement", type="primary")
with col3:
    if st.button("🔄 Reset", type="secondary"):
        st.session_state.clear()
        st.rerun()

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
                # Generate or load replacement audio (simplified for preview)
                if mode == "Text to TTS":
                    if not text_input or len(text_input.strip()) == 0:
                        st.error("Enter script text for preview.")
                    elif client is None:
                        st.error("OpenAI client not initialized.")
                    else:
                        # Generate all lines for preview
                        lines = [ln.strip() for ln in text_input.splitlines() if ln.strip()]
                        if lines:
                            st.info(f"Generating preview with {len(lines)} line(s)...")
                            combined_seg = AudioSegment.silent(duration=0)
                            pause_ms = int(line_pause * 1000)
                            
                            # Generate TTS for all lines
                            for i, line in enumerate(lines, start=1):
                                resp = client.audio.speech.create(
                                    model="gpt-4o-mini-tts",
                                    voice=voice,
                                    input=line,
                                    instructions=instructions,
                                )
                                audio_bytes = resp.read() if hasattr(resp, 'read') else resp.content
                                part_seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
                                combined_seg += part_seg
                                
                                # Add pause between lines
                                if i < len(lines) and pause_ms > 0:
                                    combined_seg += AudioSegment.silent(duration=pause_ms)
                            
                            rep_seg = combined_seg
                            
                            # Apply divine mode processing to preview
                            if divine_mode:
                                rep_wav = audiosegment_to_wav_bytes(rep_seg)
                                rep_wav = apply_serene_processing(rep_wav, brightness=brightness, divine_mode=divine_mode)
                                rep_seg = AudioSegment.from_file(io.BytesIO(rep_wav))
                            
                            # Create preview
                            preview_audio = st.session_state.preview_mixer.create_preview(
                                base_audio,
                                rep_seg,
                                int(base_spec.start * 1000),
                                int(base_spec.end * 1000),
                                crossfade_ms,
                                preview_settings['duration']
                            )
                            
                            # Export preview
                            preview_buf = io.BytesIO()
                            preview_audio.export(preview_buf, format='mp3')
                            preview_buf.seek(0)
                            
                            st.success("Preview ready!")
                            st.audio(preview_buf, format='audio/mp3')
                            
                            if preview_settings['comparison_mode']:
                                # Create before/after comparison
                                orig_preview, _ = st.session_state.preview_mixer.create_comparison_preview(
                                    base_audio,
                                    base_audio,  # Using original for now
                                    int(base_spec.start * 1000),
                                    preview_settings['duration']
                                )
                                
                                orig_buf = io.BytesIO()
                                orig_preview.export(orig_buf, format='mp3')
                                orig_buf.seek(0)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Original**")
                                    st.audio(orig_buf, format='audio/mp3')
                                with col2:
                                    st.write("**With Replacement**")
                                    st.audio(preview_buf, format='audio/mp3')
                else:
                    if not user_audio:
                        st.error("Upload replacement audio for preview.")
                    else:
                        rep_seg = AudioSegment.from_file(io.BytesIO(user_audio.read()))
                        
                        # Create preview
                        preview_audio = st.session_state.preview_mixer.create_preview(
                            base_audio,
                            rep_seg,
                            int(base_spec.start * 1000),
                            int(base_spec.end * 1000),
                            crossfade_ms,
                            preview_settings['duration']
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
                # We only extend the replacement window forward, but we must keep the original tail after the
                # original window end so that base tail content is preserved. We will first build a temp result
                # then re-append the original tail if we extended.
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
                    rep_seg = combined_seg
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

                # Auto-analyze original segment if enabled
                extracted_music = None
                if auto_analyze or enable_voice_matching:
                    st.info("🔍 Analyzing original segment characteristics...")
                    
                    # Extract and analyze original segment
                    orig_start_ms = int(base_spec.start * 1000)
                    orig_end_ms = int(base_spec.end * 1000)
                    orig_audio_np, orig_sr = extract_segment_audio(base_audio, orig_start_ms, orig_end_ms)
                    
                    # Analyze original voice profile
                    original_profile = analyze_voice_characteristics(orig_audio_np, orig_sr)
                    
                    # Extract background music/ambience
                    st.info("🎵 Extracting background music from original segment...")
                    extracted_music = st.session_state.preview_mixer.extract_music_from_original(
                        base_audio,
                        orig_start_ms,
                        orig_end_ms
                    )
                    
                    # Display analysis
                    with st.expander("📊 Original Segment Analysis", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Pitch", f"{original_profile.pitch_mean:.0f} Hz")
                            st.metric("Tempo", f"{original_profile.tempo:.0f} BPM")
                        with col2:
                            st.metric("Brightness", f"{original_profile.spectral_centroid:.0f} Hz")
                            st.metric("Dynamic Range", f"{original_profile.dynamic_range:.1f} dB")
                        with col3:
                            st.metric("Reverb Size", f"{original_profile.reverb_profile['room_size']:.2f}")
                            st.metric("Noise Floor", f"{original_profile.noise_floor:.1f} dB")
                        
                        st.info("✅ Background music extracted and will be automatically mixed with replacement")
                
                # Process replacement
                try:
                    rep_wav = audiosegment_to_wav_bytes(rep_seg)
                    
                    # Apply voice matching if enabled
                    if enable_voice_matching:
                        rep_data, rep_sr = sf.read(io.BytesIO(rep_wav))
                        if rep_data.ndim > 1:
                            rep_data = rep_data.mean(axis=1)
                        
                        # Apply matching based on selected options
                        if any([match_pitch, match_tempo, match_timbre, match_dynamics]):
                            rep_data = match_voice_profile(rep_data, rep_sr, original_profile)
                        
                        # Extract and blend ambience if requested
                        if preserve_ambience:
                            ambience = extract_ambience(orig_audio_np, orig_sr)
                            rep_data = blend_with_ambience(rep_data, ambience, mix_level=0.2)
                        
                        # Convert back to wav bytes
                        buf = io.BytesIO()
                        sf.write(buf, rep_data, rep_sr, format='WAV')
                        rep_wav = buf.getvalue()
                    
                    # Always apply de-essing to reduce harshness
                    rep_data, rep_sr = sf.read(io.BytesIO(rep_wav))
                    if rep_data.ndim > 1:
                        rep_data = rep_data.mean(axis=1)
                    
                    # Apply de-esser to reduce sibilance
                    rep_data = apply_de_esser(rep_data, rep_sr, threshold=0.6)
                    
                    # Convert back to wav bytes
                    buf = io.BytesIO()
                    sf.write(buf, rep_data, rep_sr, format='WAV')
                    rep_wav = buf.getvalue()
                    
                    # Apply standard processing with divine mode
                    rep_wav = apply_serene_processing(rep_wav, brightness=brightness, divine_mode=divine_mode)
                    rep_wav = match_loudness(orig_lufs, rep_wav)
                    rep_seg = AudioSegment.from_file(io.BytesIO(rep_wav))
                except Exception as e:
                    st.error(f"❌ Error processing replacement audio: {str(e)}")
                    st.stop()

                # Replace with looping fill; this removes the original tail inside the extended window.
                # After replacement, if we extended, re-append the original tail starting at base_spec.end.
                result_audio = replace_segment(base_audio, rep_seg, spec, crossfade_ms=crossfade_ms)
                if extend_seconds > 0:
                    # Original tail starts at the original base_spec.end timestamp
                    tail_start_ms = int(base_spec.end * 1000)
                    tail = base_audio[tail_start_ms:]
                    result_audio = result_audio + tail
                    # If tail pushes us beyond target (should not unless timing rounding), trim
                    target_ms_total = int(target_total * 1000)
                    if len(result_audio) > target_ms_total:
                        result_audio = result_audio[:target_ms_total]
                
                # Apply background music mixing (auto-use extracted music if available)
                if music_settings['enabled'] or extracted_music:
                    st.info("🎶 Mixing with background music...")
                    
                    music_audio = None
                    
                    # Use extracted music if available and no other source specified
                    if extracted_music and (not music_settings['enabled'] or music_settings['source'] == 'Extract from original'):
                        music_audio = extracted_music
                    elif music_settings['source'] == 'Extract from original' and not extracted_music:
                        # Extract music/ambience from original if not already done
                        music_audio = st.session_state.preview_mixer.extract_music_from_original(
                            base_audio, 
                            int(base_spec.start * 1000),
                            int(base_spec.end * 1000)
                        )
                    elif music_settings['source'] == 'Upload custom' and 'music_upload' in st.session_state:
                        # Use uploaded music
                        try:
                            music_file = st.session_state.get('music_upload')
                            if music_file:
                                music_audio = AudioSegment.from_file(io.BytesIO(music_file.read()))
                        except Exception as e:
                            st.warning(f"Could not load music file: {e}")
                    elif music_settings['source'] == 'Use library':
                        # Get from library
                        selected = st.session_state.get('selected_track')
                        if selected:
                            music_audio = st.session_state.music_library.get_track(selected)
                    
                    if music_audio:
                        # Apply fades to music
                        if music_settings.get('fade_in', 0) > 0:
                            music_audio = music_audio.fade_in(music_settings['fade_in'])
                        if music_settings.get('fade_out', 0) > 0:
                            music_audio = music_audio.fade_out(music_settings['fade_out'])
                        
                        # Mix with the result - use higher volume for extracted music
                        music_vol = 0.5 if extracted_music else music_settings.get('volume', 0.3)
                        
                        # Mix with the result
                        result_audio = st.session_state.preview_mixer.mix_with_background_music(
                            result_audio,
                            music_audio,
                            music_volume=music_vol,
                            ducking_enabled=music_settings.get('ducking', {}).get('enabled', True),
                            ducking_threshold=music_settings.get('ducking', {}).get('threshold', -20),
                            ducking_ratio=music_settings.get('ducking', {}).get('amount', 0.5)  # Less aggressive ducking
                        )
                        
                        st.success("✅ Background music mixed successfully")

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
