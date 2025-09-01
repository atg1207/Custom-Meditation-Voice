# Custom Meditation Voice - Project Documentation

## Overview
This is a Streamlit-based web application for replacing segments in meditation audio/video files with custom AI-generated voice or uploaded recordings. The app uses OpenAI's TTS (Text-to-Speech) API to generate natural-sounding meditation guidance and applies professional audio processing to seamlessly integrate the new segments.

## Core Features
- **Segment Replacement**: Replace specific time segments in meditation audio/video files
- **Dual Input Modes**: 
  - Text-to-Speech using OpenAI's TTS API (gpt-4o-mini-tts model)
  - Upload custom audio recordings
- **Audio Processing Pipeline**:
  - Loudness matching (LUFS normalization)
  - Serene audio processing (EQ, compression, reverb)
  - Seamless crossfading between segments
  - Loop filling for extended durations
- **Format Support**: MP4, MP3, WAV, M4A files
- **Cloud & Local**: Works on Streamlit Cloud and local environments

## Technical Architecture

### Dependencies
- **Core Framework**: Streamlit for web UI
- **Audio Processing**: 
  - pydub (main audio manipulation)
  - librosa (audio analysis)
  - pyloudnorm (LUFS loudness normalization)
  - soundfile (audio I/O)
  - pedalboard (optional: advanced effects)
- **AI/TTS**: OpenAI API
- **System**: ffmpeg (required for audio decoding)

### Project Structure
```
Custom-Meditation-Voice/
├── app.py              # Main Streamlit application
├── processing.py       # Audio processing utilities
├── requirements.txt    # Python dependencies
├── packages.txt        # System dependencies (ffmpeg)
└── The hyno guy.mp4   # Sample meditation video
```

### Key Components

#### app.py (Main Application)
- **UI Components**:
  - File upload/selection
  - Time segment specification (start/end timestamps)
  - TTS configuration (voice, style, pauses)
  - Audio processing parameters
- **Processing Flow**:
  1. Load source audio/video
  2. Generate or load replacement audio
  3. Apply audio processing
  4. Replace segment with crossfading
  5. Export processed file

#### processing.py (Audio Processing)
- **SegmentSpec**: Data class for segment timing
- **Audio I/O**: Functions for loading/exporting AudioSegment
- **Loudness Processing**:
  - `compute_loudness()`: Calculate LUFS
  - `match_loudness()`: Normalize to target LUFS
- **Effects Processing**:
  - `apply_serene_processing()`: Apply EQ, compression, reverb
  - Fallback mode when pedalboard unavailable
- **Segment Operations**:
  - `loop_to_duration()`: Loop audio to fill duration
  - `replace_segment()`: Replace with crossfading

### Configuration & Settings

#### TTS Parameters
- **Voices**: 11 OpenAI voices (alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse)
- **Style Instructions**: Customizable voice style prompt
- **Line Pauses**: Configurable pause between script lines (0-5 seconds)

#### Audio Processing Settings
- **Brightness**: 0.0-1.0 (affects EQ and reverb)
- **Crossfade**: 50-1000ms transition duration
- **Loudness Target**: -30 to -10 LUFS

## Improvement Opportunities

### 1. User Experience Enhancements
- **Real-time Preview**: Add audio preview before final processing
- **Waveform Visualization**: Display audio waveform with segment markers
- **Batch Processing**: Support multiple segment replacements
- **Preset Templates**: Save/load common TTS and processing settings
- **Progress Indicators**: More detailed progress for long operations

### 2. Audio Processing Improvements
- **Advanced Crossfading**: Multiple crossfade algorithms (linear, exponential, S-curve)
- **Noise Reduction**: Integrate noisereduce library (already in requirements)
- **Dynamic Range Control**: More sophisticated compression/limiting
- **Spectral Matching**: Match frequency characteristics of original audio
- **Ambience Extraction**: Preserve background ambience during replacement

### 3. TTS & Voice Features
- **Multiple TTS Providers**: Support for ElevenLabs, Azure, Google TTS
- **Voice Cloning**: Custom voice training/cloning capabilities
- **SSML Support**: Advanced speech markup for better control
- **Multilingual Support**: Support for multiple languages
- **Emotion Control**: Fine-grained emotional expression parameters

### 4. Performance Optimizations
- **Caching System**: Cache TTS results to avoid regeneration
- **Chunked Processing**: Process large files in chunks
- **Parallel Processing**: Multi-threaded audio processing
- **Memory Management**: Stream processing for large files
- **GPU Acceleration**: Use CUDA for audio processing where available

### 5. File Management
- **Cloud Storage Integration**: S3, Google Drive, Dropbox support
- **Project Management**: Save/load complete projects
- **Version Control**: Track changes and allow rollback
- **Export Options**: Multiple format/quality export options

### 6. Advanced Features
- **AI Script Generation**: Generate meditation scripts using LLMs
- **Music Integration**: Background music mixing capabilities
- **Binaural Beats**: Add therapeutic frequency generation
- **Time Stretching**: Adjust segment duration without pitch change
- **Automatic Segmentation**: AI-based detection of suitable replacement points

### 7. Developer Experience
- **API Mode**: REST API for programmatic access
- **Plugin System**: Extensible architecture for custom processors
- **Testing Suite**: Comprehensive unit and integration tests
- **Documentation**: API documentation and usage examples
- **Docker Support**: Containerized deployment

### 8. Error Handling & Reliability
- **Robust Error Recovery**: Better handling of corrupted files
- **Input Validation**: Comprehensive validation of user inputs
- **Fallback Mechanisms**: Graceful degradation when services unavailable
- **Logging System**: Detailed logging for debugging
- **Health Checks**: System status monitoring

## Environment Setup

### Local Development
1. Install Python 3.8+
2. Install ffmpeg: `apt-get install ffmpeg` (Linux) or `brew install ffmpeg` (Mac)
3. Install dependencies: `pip install -r requirements.txt`
4. Set OpenAI API key in `.env` file: `OPENAI_API_KEY=your-key-here`
5. Run: `streamlit run app.py`

### Streamlit Cloud Deployment
1. Add `OPENAI_API_KEY` to Streamlit secrets
2. ffmpeg automatically installed via packages.txt
3. Deploy directly from GitHub repository

## API Keys & Security
- OpenAI API key required for TTS functionality
- Supports both environment variables and Streamlit secrets
- No keys or sensitive data stored in code
- Secure API key handling with proper error messages

## Current Limitations
- Single segment replacement per operation
- Limited to OpenAI TTS voices
- No real-time preview capability
- Basic crossfading algorithm only
- No undo/redo functionality