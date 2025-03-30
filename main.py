# -*- coding: utf-8 -*-
import sys
import os
import re
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QProgressBar, QTextEdit, QFileDialog, QMessageBox,
    QComboBox, QSlider, QStyleFactory
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QColor, QPalette
from dotenv import load_dotenv

from pytube import Playlist, YouTube
from pytube.exceptions import PytubeError
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# --- Configuration Loading & Constants ---
load_dotenv(".env")

# Style Constants
COLOR_BACKGROUND = "#2c3e50"
COLOR_WIDGET_BACKGROUND = "#34495e"
COLOR_BORDER = "#3498db"
COLOR_TEXT = "#ecf0f1"
COLOR_ACCENT1 = "#2ecc71"
COLOR_ACCENT2 = "#e74c3c"
COLOR_ACCENT3 = "#95a5a6" # Disabled color
COLOR_PLACEHOLDER = "#7f8c8d"
FONT_FAMILY = "Segoe UI"

# Default values / Fallbacks
DEFAULT_CHUNK_SIZE = 3000
MIN_CHUNK_SIZE = 500
MAX_CHUNK_SIZE = 50000
DEFAULT_MODEL = "gemini-2.5-pro-exp-03-25"
DEFAULT_CATEGORY_NAME = "Balanced and Detailed"
DEFAULT_CONFIG_FILE = "config.json"
DEFAULT_TRANSCRIPT_LANGUAGES = ['en']
DEFAULT_OUTPUT_DIR_FALLBACK = os.path.expanduser("~")

# --- Fallback Configuration (if config.json is missing/invalid) ---
DEFAULT_PROMPTS_CONFIG: Dict[str, Dict[str, Any]] = {
    "Balanced and Detailed": {
        "prompt": """Please refine the following YouTube transcript text into a well-structured, readable format using Markdown.
Focus on clarity, accuracy, and capturing the main points. Correct obvious transcription errors if possible.
Maintain the original speaker's intent and tone where appropriate. Use headings, bullet points, or numbered lists for structure.
Ensure the output is primarily in [Language].

Transcript Text:""",
        "chunk_size": 4000
    },
    "Concise Summary": {
        "prompt": """Generate a concise summary of the key points from the following YouTube transcript text using Markdown bullet points.
Focus only on the most important information and conclusions.
Ensure the output is primarily in [Language].

Transcript Text:""",
        "chunk_size": 10000
    },
    "Technical Documentation": {
        "prompt": """Convert the following technical YouTube transcript into structured documentation using Markdown.
Use headings for sections, code blocks for code examples (if any), and bullet points for steps or features.
Prioritize technical accuracy and clarity. Format it for easy reference.
Ensure the output is primarily in [Language].

Transcript Text:""",
        "chunk_size": 3500
    }
}

# --- Logging Configuration ---
log_file = 'youtube_gemini_processor.log'
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(module)s.%(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def get_video_id_from_url(url: str) -> Optional[str]:
    """Extracts YouTube video ID from various URL formats."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/|v\/|youtu\.be\/)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    logger.debug(f"Could not extract video ID from URL: {url}")
    return None

def load_config(config_file: str) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    """Loads prompts and default output directory from a JSON config file."""
    prompts_config: Dict[str, Dict[str, Any]] = {}
    default_output_dir: Optional[str] = None
    config_data: Dict[str, Any] = {}

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        if not isinstance(config_data, dict):
             raise ValueError("Config file is not a valid JSON object.")

        # Extract prompts - robust check
        prompts_section = config_data.get("prompts", {})
        if isinstance(prompts_section, dict) and all(
             isinstance(v, dict) and 'prompt' in v and 'chunk_size' in v
             for v in prompts_section.values()
        ):
             prompts_config = prompts_section
             logger.info(f"Successfully loaded prompts configuration from {config_file}")
        else:
             logger.warning(f"'prompts' section in '{config_file}' is missing or invalid. Using default prompts.")
             prompts_config = DEFAULT_PROMPTS_CONFIG

        # Extract default output directory
        loaded_dir = config_data.get("default_output_directory")
        if isinstance(loaded_dir, str) and loaded_dir:
             default_output_dir = loaded_dir
             logger.info(f"Loaded default output directory: {default_output_dir}")
        else:
             logger.info("No valid 'default_output_directory' found in config. Using fallback.")
             default_output_dir = None

    except FileNotFoundError:
        logger.warning(f"Config file '{config_file}' not found. Using default prompts and no default directory.")
        prompts_config = DEFAULT_PROMPTS_CONFIG
        default_output_dir = None
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error loading or parsing '{config_file}': {e}. Using defaults.")
        prompts_config = DEFAULT_PROMPTS_CONFIG
        default_output_dir = None
    except Exception as e:
        logger.exception(f"Unexpected error loading config '{config_file}': {e}. Using defaults.")
        prompts_config = DEFAULT_PROMPTS_CONFIG
        default_output_dir = None

    # Ensure prompts_config isn't empty
    if not prompts_config:
         logger.critical("CRITICAL: No prompts configuration loaded, falling back to hardcoded defaults.")
         prompts_config = DEFAULT_PROMPTS_CONFIG

    return prompts_config, default_output_dir

def save_config(prompts_config: Dict[str, Dict[str, Any]], default_output_dir: Optional[str], config_file: str):
    """Saves the current configuration (prompts and default dir) to the JSON file."""
    config_data = {
        "prompts": prompts_config,
        "default_output_directory": default_output_dir if isinstance(default_output_dir, str) and default_output_dir else None
    }
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Configuration successfully saved to {config_file}")
        return True
    except (IOError, TypeError) as e:
        logger.error(f"Error saving configuration to '{config_file}': {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error saving config '{config_file}': {e}")
        return False

# --- Threads ---

class TranscriptExtractionThread(QThread):
    """
    Worker thread to extract transcripts from YouTube videos or playlists.
    Fetches transcripts using YouTubeTranscriptApi and saves them to an
    intermediate plain text file.
    """
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    extraction_complete = pyqtSignal(str) # Emits path to intermediate .txt file
    error_occurred = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, url: str, output_file: str,
                 preferred_languages: List[str], # Changed from optional with default
                 parent=None):
        super().__init__(parent)
        self.url = url
        self.output_file = output_file
        # Ensure we store a valid list, even if passed None/empty somehow
        self.preferred_languages = preferred_languages if preferred_languages else DEFAULT_TRANSCRIPT_LANGUAGES
        self._is_running = True
        logger.info(f"TranscriptExtractionThread initialized for URL: {url}, Languages: {self.preferred_languages}, Output: {self.output_file}")

    def run(self):
        """Executes the transcript extraction process."""
        logger.info(f"Starting transcript extraction to intermediate file: {self.output_file}")
        self.status_update.emit(f"Extracting transcripts to intermediate file: {os.path.basename(self.output_file)}...")
        try:
            video_urls: List[str] = []
            playlist_title: Optional[str] = None
            is_playlist = "list=" in self.url and ("youtube.com/" in self.url or "youtu.be/" in self.url)

            # Determine if URL is a playlist or single video
            if is_playlist:
                self.status_update.emit("Fetching playlist information...")
                logger.info(f"Processing as playlist: {self.url}")
                try:
                    pl = Playlist(self.url)
                    playlist_title = pl.title
                    # Ensure video_urls is populated correctly
                    if not pl.video_urls:
                        raise PytubeError(f"Playlist object created but video_urls list is empty for {self.url}")
                    video_urls = list(pl.video_urls) # Convert generator to list
                    if not video_urls:
                         err_msg = f"Playlist found but contains no videos (or failed to fetch URLs): {self.url}"
                         self.error_occurred.emit(err_msg)
                         logger.warning(err_msg)
                         self.finished_signal.emit()
                         return
                    self.status_update.emit(f"Found playlist '{playlist_title}' with {len(video_urls)} videos.")
                    logger.info(f"Playlist '{playlist_title}' found with {len(video_urls)} videos.")
                except PytubeError as e:
                    err_msg = f"Error accessing playlist (check URL/network/video availability): {e}"
                    self.error_occurred.emit(err_msg)
                    logger.error(err_msg)
                    self.finished_signal.emit()
                    return
                except Exception as e:
                    err_msg = f"Unexpected error fetching playlist details: {e}"
                    self.error_occurred.emit(err_msg)
                    logger.exception("Unexpected error during playlist fetch:")
                    self.finished_signal.emit()
                    return

            elif get_video_id_from_url(self.url):
                video_id = get_video_id_from_url(self.url)
                logger.info(f"Processing as single video: {self.url}")
                video_urls = [self.url]
                try:
                    yt = YouTube(self.url)
                    playlist_title = f"Single Video: {yt.title}" # Use actual title if possible
                except Exception as e:
                    logger.warning(f"Could not fetch single video title for '{self.url}': {e}")
                    playlist_title = f"Single Video ({video_id})"
                self.status_update.emit("Processing single video.")
            else:
                err_msg = "Invalid URL: Doesn't appear to be a valid YouTube video or playlist URL."
                self.error_occurred.emit(err_msg)
                logger.warning(f"Invalid URL passed to extraction thread: {self.url}")
                self.finished_signal.emit()
                return

            total_videos = len(video_urls)
            processed_count = 0
            error_count = 0

            # Check write permissions for intermediate file directory (usually script dir)
            output_dir = os.path.dirname(self.output_file)
            if not output_dir: output_dir = "." # Handle case where intermediate file is in current dir

            if not os.path.exists(output_dir):
                 try:
                    os.makedirs(output_dir, exist_ok=True)
                    logger.info(f"Created intermediate file directory: {output_dir}")
                 except OSError as e:
                    err_msg = f"Could not create intermediate file directory '{output_dir}': {e}"
                    self.error_occurred.emit(err_msg); logger.error(err_msg)
                    self.finished_signal.emit(); return
            elif not os.access(output_dir, os.W_OK):
                 err_msg = f"Cannot write to intermediate file directory '{output_dir}'. Check permissions."
                 self.error_occurred.emit(err_msg); logger.error(err_msg)
                 self.finished_signal.emit(); return

            # Write transcripts to intermediate file
            try:
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Intermediate Transcript Data (Plain Text)\n")
                    f.write(f"# Auto-generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    if playlist_title:
                        f.write(f"Source Title: {playlist_title}\n")
                        f.write(f"Source URL: {self.url}\n\n")

                    for index, video_url in enumerate(video_urls):
                        if not self._is_running:
                            self.status_update.emit("Transcript extraction cancelled.")
                            logger.info("Transcript extraction loop cancelled.")
                            break

                        self.status_update.emit(f"Processing video {index + 1}/{total_videos}: {video_url}")
                        logger.info(f"Attempting transcript fetch for video {index + 1}/{total_videos}: {video_url}")
                        video_id = get_video_id_from_url(video_url)

                        f.write(f"--- Video Start ---\n")
                        f.write(f"Video URL: {video_url}\n")

                        if not video_id:
                            msg = f"Could not extract video ID from URL: {video_url}"
                            self.status_update.emit(f"<font color='{COLOR_ACCENT2}'>{msg}</font>")
                            logger.warning(f"Failed to extract video ID: {video_url}")
                            error_count += 1
                            f.write(f"Transcript: [Error - Could not extract Video ID]\n")
                        else:
      
# --- START: REPLACEMENT BLOCK for Transcript Fetching ---
                            try:
                                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                                transcript = None
                                languages_tried = []
                                found_lang_details = "None" # For logging

                                # 1. Try preferred languages provided by the user
                                try:
                                    if self.preferred_languages:
                                        # Make a copy to avoid modifying the original list if needed elsewhere
                                        current_preference = list(self.preferred_languages)
                                        languages_tried.extend(current_preference)
                                        logger.debug(f"Attempting to find transcript in preferred languages: {current_preference} for {video_id}")
                                        # find_transcript tries languages in the order given
                                        transcript = transcript_list.find_transcript(current_preference)
                                        found_lang_details = f"{transcript.language} ({'auto-generated' if transcript.is_generated else 'manual'})"
                                        logger.info(f"Found user-preferred transcript: {found_lang_details} for {video_id}")
                                except NoTranscriptFound:
                                    logger.warning(f"No transcript found in preferred languages {self.preferred_languages} for {video_id}.")
                                    transcript = None # Explicitly set to None

                                # 2. If no preferred transcript found, try fallback to generated English ('a.en')
                                # Check if 'a.en' (or just 'en' if preferred is just 'en') wasn't already tried or found
                                fallback_lang_code = 'en' # Target 'en' for generated transcripts
                                needs_fallback_check = False
                                if not transcript:
                                    # Check if 'en' (case-insensitive) was among the preferred languages
                                    if fallback_lang_code not in [lang.lower() for lang in self.preferred_languages]:
                                     needs_fallback_check = True
                                    # Even if 'en' was preferred, maybe only manual 'en' existed and generated wasn't tried
                                    elif transcript is None: # Double check needed if 'en' was preferred but find_transcript failed
                                        needs_fallback_check = True


                                if needs_fallback_check:
                                    try:
                                        # Use find_generated_transcript for specifically getting auto-generated ones
                                        logger.debug(f"Attempting fallback to generated English ('{fallback_lang_code}') for {video_id}")
                                        transcript = transcript_list.find_generated_transcript([fallback_lang_code])
                                        # Update details if fallback succeeded
                                        found_lang_details = f"{transcript.language} (Generated Fallback)"
                                        languages_tried.append(f"{fallback_lang_code}-generated") # Mark fallback attempt
                                        logger.info(f"Found fallback generated English transcript ({transcript.language}) for {video_id}")
                                    except NoTranscriptFound:
                                        logger.warning(f"Fallback to generated English ('{fallback_lang_code}') also failed for {video_id}.")
                                        transcript = None # Fallback failed

                                # 3. Process if a transcript was found (either preferred or fallback)
                                if transcript:
                                    transcript_data = transcript.fetch()
                                    # Join text parts; fetch returns list of dicts {'text': '...', 'start': ..., 'duration': ...}
                                    transcript_text = ' '.join([entry.text for entry in transcript_data])

                                    
                                    f.write(f"Transcript:\n{transcript_text}\n")
                                    processed_count += 1
                                    logger.info(f"Successfully fetched and wrote transcript for video {index + 1}: {video_id}")
                                else:
                                    # No transcript found after all attempts for this video_id
                                    available_langs_list = [t.language for t in transcript_list]
                                    available_langs = ", ".join(available_langs_list) if available_langs_list else "None Available"
                                    unique_tried = sorted(list(set(languages_tried))) # Show unique attempts
                                    msg = (f"Transcript not found (Tried: {unique_tried}; "
                                       f"Available: [{available_langs}]) for video {index + 1}: {video_url}")
                                    self.status_update.emit(f"<font color='#f39c12'>{msg}</font>")
                                    logger.warning(msg)
                                    f.write(f"Transcript: [Not Available in {unique_tried} or Disabled]\n")
                                    error_count += 1

                            except TranscriptsDisabled:
                                msg = f"Transcripts are disabled for video {index + 1}: {video_url}"
                                self.status_update.emit(f"<font color='#f39c12'>{msg}</font>")
                                logger.warning(msg)
                                f.write(f"Transcript: [Transcripts Disabled]\n")
                                error_count += 1
                            except PytubeError as pe: # Catch Pytube specific errors if they leak from listing
                                msg = f"Pytube error accessing video info {index + 1} ({video_url}): {pe}"
                                self.status_update.emit(f"<font color='{COLOR_ACCENT2}'>{msg}</font>")
                                logger.error(f"Pytube error for {video_id}: {pe}", exc_info=True)
                                f.write(f"Transcript: [Error Accessing Video - {pe}]\n")
                                error_count += 1
                            except Exception as video_error:
                                # Catch other potential errors during listing/finding/fetching
                                msg = f"Error processing video {index + 1} ({video_url}): {video_error}"
                                self.status_update.emit(f"<font color='{COLOR_ACCENT2}'>{msg}</font>")
                                logger.error(f"Error fetching/processing transcript for {video_id}: {video_error}", exc_info=True)
                                f.write(f"Transcript: [Error Processing - {video_error}]\n")
                                error_count += 1
                        # --- END: REPLACEMENT BLOCK ---
                        f.write(f"--- Video End ---\n\n")

                        # Update progress bar
                        progress_percent = int(((index + 1) / total_videos) * 100)
                        self.progress_update.emit(progress_percent)

            except IOError as e:
                 err_msg = f"Error writing intermediate transcript file '{self.output_file}': {e}"
                 self.error_occurred.emit(err_msg)
                 logger.error(err_msg)

            # Final status update
            if not self._is_running:
                 final_msg = f"Intermediate transcript extraction CANCELLED. Processed: {processed_count}, Skipped/Errors: {error_count} before stopping."
                 self.status_update.emit(f"<font color='#e67e22'>{final_msg}</font>")
                 logger.info(final_msg)
            elif processed_count > 0:
                 final_msg = f"Intermediate transcript extraction finished. Processed: {processed_count}, Skipped/Errors: {error_count}."
                 self.status_update.emit(final_msg)
                 logger.info(final_msg)
                 self.extraction_complete.emit(self.output_file) # Emit the path
            elif total_videos > 0 and error_count == total_videos:
                 err_msg = "Extraction finished, but failed to get transcripts for any video."
                 self.error_occurred.emit(err_msg)
                 logger.warning(err_msg)
            elif total_videos == 0 and not is_playlist: # Should be caught earlier, but safety check
                 err_msg = "Input URL did not yield any videos to process."
                 self.error_occurred.emit(err_msg)
                 logger.warning(err_msg)

        except Exception as e:
            err_msg = f"Unexpected error during transcript extraction: {e}"
            self.error_occurred.emit(err_msg)
            logger.exception("Unhandled exception in TranscriptExtractionThread:")
        finally:
            logger.debug("TranscriptExtractionThread run() method finished.")
            self.finished_signal.emit()

    def stop(self):
        """Requests the thread to stop processing."""
        self.status_update.emit("Cancellation requested...")
        logger.info("Stop requested for TranscriptExtractionThread.")
        self._is_running = False


class GeminiProcessingThread(QThread):
    """
    Worker thread to process transcripts using the Gemini API.
    Reads the intermediate transcript file, splits content into chunks,
    sends chunks to the specified Gemini model with a prompt, and saves
    the refined output to a final Markdown file.
    """
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    processing_complete = pyqtSignal(str) # Emits path to final .md file
    error_occurred = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, intermediate_transcript_file: str, final_output_file_md: str, api_key: str,
                 model_name: str, output_language: str, chunk_size: int,
                 prompt_template: str, style_category: str, # Added parameter
                 parent=None):
        super().__init__(parent)
        self.input_file = intermediate_transcript_file
        self.output_file = final_output_file_md
        self.api_key = api_key
        self.model_name = model_name
        self.output_language = output_language
        self.chunk_size = max(chunk_size, MIN_CHUNK_SIZE)
        self.prompt_template = prompt_template
        self.style_category = style_category # Added this line
        self._is_running = True
        self._model = None
        logger.info(f"GeminiProcessingThread initialized for input: {self.input_file}, output: {self.output_file}")
        logger.info(f"Model: {model_name}, Lang: {output_language}, Chunk Size: {self.chunk_size}, Style: {self.style_category}") # Added style to log

    def _initialize_gemini(self) -> bool:
        """Configures the Gemini API and initializes the model."""
        if not self.api_key:
            self.error_occurred.emit("Gemini API Error: API key is missing.")
            logger.error("Gemini API key is missing.")
            return False
        try:
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini API configured successfully for model {self.model_name}.")
            return True
        except google_exceptions.PermissionDenied:
            err_msg = "Gemini API Error: Permission Denied. Check API key validity and permissions."
            self.error_occurred.emit(err_msg); logger.error(err_msg); return False
        except google_exceptions.NotFound:
             err_msg = f"Gemini API Error: Model '{self.model_name}' not found or access denied."
             self.error_occurred.emit(err_msg); logger.error(err_msg); return False
        except google_exceptions.InvalidArgument as e:
             # Often indicates a malformed API key
            err_msg = f"Gemini API Error: Invalid Argument. (Is API Key correctly formatted?). Details: {e}"
            self.error_occurred.emit(err_msg); logger.error(err_msg); return False
        except Exception as e:
            err_msg = f"Failed to initialize Gemini API: {e}"
            self.error_occurred.emit(err_msg); logger.exception("Gemini API initialization failed:"); return False

    def _split_transcript_into_videos(self, file_path: str) -> List[Tuple[str, str]]:
        """Parses the intermediate file content into (URL, transcript) tuples."""
        videos = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except IOError as e:
            err_msg = f"Error reading intermediate transcript file '{file_path}': {e}"
            self.error_occurred.emit(err_msg); logger.error(err_msg); return []

        # Regex to find video blocks in the intermediate file
        pattern = re.compile(
            r"^--- Video Start ---\s*^Video URL:\s*(.*?)\s*^Transcript:\s*(.*?)\s*^--- Video End ---",
            re.MULTILINE | re.DOTALL
        )
        skipped_marker = "[Not Available"
        error_marker = "[Error"
        for match in pattern.finditer(content):
            url = match.group(1).strip()
            transcript = match.group(2).strip()
            # Only process videos with actual transcript text, skipping errors/unavailable
            if transcript and not transcript.startswith(skipped_marker) and not transcript.startswith(error_marker):
                videos.append((url, transcript))
            else:
                logger.info(f"Skipping Gemini processing for unavailable/error video: {url}")

        if not videos:
            logger.warning(f"No processable transcripts found in intermediate file: {file_path}.")
        else:
            logger.info(f"Split intermediate file into {len(videos)} processable video transcripts.")
        return videos

    def _split_text_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Splits a large text into chunks of approximately chunk_size words."""
        words = text.split()
        chunks = []
        if not words:
            return []

        current_chunk_start = 0
        while current_chunk_start < len(words):
            end_index = min(current_chunk_start + chunk_size, len(words))
            chunks.append(" ".join(words[current_chunk_start:end_index]))
            current_chunk_start = end_index

        # Avoid tiny final chunks: merge if smaller than 10% of chunk size or 100 words
        min_merge_threshold = max(100, chunk_size // 10)
        if len(chunks) > 1 and len(chunks[-1].split()) < min_merge_threshold:
            logger.debug(f"Merging small final chunk (size {len(chunks[-1].split())} words) into previous chunk.")
            chunks[-2] += " " + chunks.pop()

        logger.debug(f"Split text ({len(words)} words) into {len(chunks)} chunks for Gemini processing.")
        return chunks

    def run(self):
        """Executes the Gemini processing workflow."""
        if not self._initialize_gemini():
            self.finished_signal.emit(); return

        video_data = self._split_transcript_into_videos(self.input_file)
        if not video_data:
            # Error message emitted by _split_transcript_into_videos if file read fails
            if os.path.exists(self.input_file):
                 err_msg = "No processable video transcripts found in the intermediate file."
                 self.error_occurred.emit(err_msg); logger.warning(f"{err_msg} Path: {self.input_file}")
            # else: Intermediate file not found error already handled
            self.finished_signal.emit(); return

        total_videos = len(video_data); processed_video_count = 0

        # Check write permissions for final output directory
        output_dir = os.path.dirname(self.output_file)
        if not output_dir: output_dir = "."

        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Created final output directory: {output_dir}")
            except OSError as e:
                err_msg = f"Could not create final output directory '{output_dir}': {e}"
                self.error_occurred.emit(err_msg); logger.error(err_msg); self.finished_signal.emit(); return
        elif not os.access(output_dir, os.W_OK):
             err_msg = f"Cannot write to final output directory '{output_dir}'. Check permissions."
             self.error_occurred.emit(err_msg); logger.error(err_msg); self.finished_signal.emit(); return

        # Use a temporary file for accumulating streamed responses per video
        temp_response_file_path = self.output_file + ".gemini_temp"

        try:
            # --- <<< MODIFIED HEADER WRITING >>> ---
            # Write YAML front matter to the final output file
            with open(self.output_file, "w", encoding="utf-8") as final_f:
                current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                final_f.write("---\n")
                final_f.write(f"processing_date: {current_datetime}\n") # Renamed field
                final_f.write(f"gemini_model: {self.model_name}\n")     # Renamed field
                final_f.write(f"refinement_style: {self.style_category}\n") # Added style category
                final_f.write(f"output_language: {self.output_language}\n") # Added output language
                # Add any other global metadata here if desired (e.g., source_url if only one video)
                final_f.write("---\n\n") # End YAML block
            # --- <<< END OF MODIFIED HEADER WRITING >>> ---

            # Process each video transcript (Rest of the loop remains the same)
            for video_index, (video_url, video_transcript) in enumerate(video_data):
                if not self._is_running:
                    self.status_update.emit("Gemini processing cancelled.")
                    logger.info("Gemini processing loop cancelled.")
                    break

                self.status_update.emit(f"\n<font color='{COLOR_ACCENT1}'>Processing Video {video_index + 1}/{total_videos}:</font> {video_url}")
                logger.info(f"Starting Gemini processing for video {video_index + 1}/{total_videos}: {video_url}")
                word_count = len(video_transcript.split())
                self.status_update.emit(f"  Transcript length: ~{word_count} words")

                video_transcript_chunks = self._split_text_into_chunks(video_transcript, self.chunk_size)
                if not video_transcript_chunks:
                    logger.warning(f"Video {video_index + 1} ({video_url}) resulted in zero chunks after splitting, skipping.")
                    continue
                self.status_update.emit(f"  Processing in {len(video_transcript_chunks)} chunk(s)...")

                video_succeeded = True
                stop_video_processing = False # Flag to break outer loop if needed

                # Process chunks for the current video
                try:
                    with open(temp_response_file_path, "w", encoding="utf-8") as temp_f:
                        previous_response_context = ""
                        for chunk_index, chunk in enumerate(video_transcript_chunks):
                             # ... (Inner chunk processing loop - No changes needed here) ...
                             if not self._is_running:
                                logger.info(f"Gemini chunk processing cancelled during video {video_index + 1}.")
                                video_succeeded = False; stop_video_processing = True; break # Break chunk loop

                             status_msg = (f"  Video {video_index + 1}, Chunk {chunk_index + 1}/{len(video_transcript_chunks)}: Sending request...")
                             self.status_update.emit(status_msg); logger.debug(status_msg)

                             formatted_prompt_core = self.prompt_template.replace("[Language]", self.output_language)
                             context_prefix = ""
                             if previous_response_context and chunk_index > 0:
                                context_limit = 500
                                context_preview = previous_response_context[-context_limit:]
                                context_prefix = (
                                    f"This is a continuation of previous processing.\n"
                                    f"Context from end of previous part:\n...\n{context_preview}\n\n"
                                    "Now continue processing the following transcript text:\n"
                                )
                             full_prompt = f"{context_prefix}{formatted_prompt_core}\n\n{chunk}"
                             generated_text_for_chunk = ""
                             # --- START: REPLACEMENT BLOCK for API Call & Retry ---
                             full_prompt = f"{context_prefix}{formatted_prompt_core}\n\n{chunk}"
                             generated_text_for_chunk = ""
                             max_retries = 3 # Number of retries for rate limit errors
                             retry_delay_seconds = 5 # Initial delay in seconds

                             # --- Retry Loop ---
                             for attempt in range(max_retries):
                                 try:
                                     if not self._model: raise RuntimeError("Gemini model not initialized before use.")

                                     # Make the API call
                                     logger.debug(f"Attempt {attempt + 1}/{max_retries}: Sending chunk {chunk_index + 1} (Video {video_index + 1}) to Gemini.")
                                     response = self._model.generate_content(full_prompt, stream=True)
                                     self.status_update.emit(f"  Video {video_index + 1}, Chunk {chunk_index + 1}: Receiving stream (Attempt {attempt+1})...")
                                       # Process the stream response
                                     temp_generated_text = "" # Accumulate text for this attempt
                                     for response_chunk in response:
                                          if not self._is_running:
                                              logger.info(f"Cancellation detected during stream for video {video_index + 1}, chunk {chunk_index+1}.")
                                              video_succeeded = False; stop_video_processing = True
                                              raise InterruptedError("Processing cancelled during stream") # Raise to break loops

                                          # Handle potential errors getting text from stream parts
                                          try:
                                              chunk_text = response_chunk.text
                                          except ValueError:
                                              logger.debug("Received an empty value part in the stream, skipping.")
                                              continue # Skip this part of the stream
                                          except Exception as e_stream_part:
                                              logger.warning(f"Error accessing text in stream part: {e_stream_part}", exc_info=False) # Log less verbosely
                                              chunk_text = "" # Treat as empty

                                          if chunk_text:
                                              temp_f.write(chunk_text); temp_f.flush()
                                              temp_generated_text += chunk_text

                                     # If the stream finished without error for this attempt, store result and break retry loop
                                     generated_text_for_chunk = temp_generated_text
                                     logger.debug(f"Successfully processed chunk {chunk_index + 1} on attempt {attempt + 1}")
                                     video_succeeded = True # Mark as success for this chunk
                                     break # Exit the retry loop

                                 # --- Exception Handling for the API call ---
                                 except InterruptedError: # Propagate cancellation immediately
                                     raise
                                 except google_exceptions.ResourceExhausted as rate_limit_err:
                                     logger.warning(f"Rate limit hit on attempt {attempt + 1}/{max_retries} for chunk {chunk_index+1}: {rate_limit_err}. Retrying in {retry_delay_seconds}s...")
                                     self.status_update.emit(f"<font color='#f39c12'>Rate limit hit, retrying chunk {chunk_index+1} (attempt {attempt+1}/{max_retries})...</font>")
                                     if attempt + 1 == max_retries:
                                         logger.error(f"Max retries ({max_retries}) reached for rate limit on chunk {chunk_index+1}.")
                                         # Emit specific error and mark failure
                                         self.error_occurred.emit(f"Gemini API Error: Rate limit exceeded after {max_retries} retries. Consider increasing chunk size or wait.")
                                         video_succeeded = False; stop_video_processing = True
                                         raise # Re-raise the last exception to break outer loops
                                     # Wait before retrying
                                         time.sleep(retry_delay_seconds)
                                         retry_delay_seconds *= 2 # Exponential backoff (wait longer next time)
                                 except google_exceptions.GoogleAPIError as api_err:
                                       # Handle other non-retryable Google API errors
                                     err_msg = f"Gemini API error video {video_index+1}, chunk {chunk_index+1} (Attempt {attempt+1}): {api_err}"
                                     self.status_update.emit(f"<font color='{COLOR_ACCENT2}'>{err_msg}</font>")
                                     logger.error(err_msg, exc_info=True)
                                     temp_f.write(f"\n\n[API Error: {api_err}]\n") # Write error marker to temp file
                                     video_succeeded = False; stop_video_processing = True
                                     raise # Re-raise to break outer loops
                                 except Exception as chunk_err:
                                     # Handle other unexpected errors during API call or stream processing
                                     err_msg = f"Unexpected error processing chunk {chunk_index+1} video {video_index+1} (Attempt {attempt+1}): {chunk_err}"
                                     self.status_update.emit(f"<font color='{COLOR_ACCENT2}'>{err_msg}</font>")
                                     logger.exception(f"Unexpected error processing chunk {chunk_index+1} video {video_index+1}:")
                                     temp_f.write(f"\n\n[Processing Error: {chunk_err}]\n") # Write error marker
                                     video_succeeded = False; stop_video_processing = True
                                     raise # Re-raise to break outer loops
                               # --- End Retry Loop ---

                               # Check if we need to stop processing this video due to errors/cancellation
                                 if not video_succeeded or stop_video_processing:
                                   logger.warning(f"Aborting further chunk processing for video {video_index + 1} due to error or cancellation.")
                                   break # Exit the chunk loop for this video

                               # If we successfully processed the chunk (exited retry loop via break)
                                 if not generated_text_for_chunk and video_succeeded:
                                    # Log warning if Gemini returned nothing for the chunk
                                    logger.warning(f"Gemini returned an empty response for video {video_index+1}, chunk {chunk_index+1}.")
                                    self.status_update.emit(f"<font color='#f39c12'>Warning: Empty response for chunk {chunk_index+1}.</font>")

                               # Only update context and success status if the chunk loop wasn't broken by error/cancel
                                 if video_succeeded:
                                   previous_response_context = generated_text_for_chunk # Use text from successful attempt
                                   self.status_update.emit(f"  Video {video_index + 1}, Chunk {chunk_index + 1} processed.")
                                   logger.debug(f"Gemini processing finished successfully for chunk {chunk_index + 1}.")
                               # --- END: REPLACEMENT BLOCK ---

                        if not video_succeeded:
                            logger.warning(f"Aborting further chunk processing for video {video_index + 1} due to error or cancellation.")

                except IOError as e:
                     err_msg = f"Error writing temp Gemini output file '{temp_response_file_path}' for video {video_index + 1}: {e}"
                     self.status_update.emit(f"<font color='{COLOR_ACCENT2}'>{err_msg}</font>")
                     logger.error(err_msg); video_succeeded = False
                # --- End temp file handling for video ---

                # Append results for this video (or error marker) to the final file
                if not stop_video_processing:
                    try:
                        with open(self.output_file, "a", encoding="utf-8") as final_f:
                            # Use HTML comments for machine-readable markers (less likely to clash with content)
                            final_f.write(f"<!-- Processed Video Start: {video_index + 1} -->\n")
                            final_f.write(f"<!-- Original URL: {video_url} -->\n\n")

                            if video_succeeded:
                                try:
                                    # Read the raw output from the temp file for this video
                                    with open(temp_response_file_path, "r", encoding="utf-8") as temp_f_read:
                                        gemini_output_raw = temp_f_read.read()

                                    # --- Optional Post-processing to strip common intro phrases ---
                                    gemini_output_processed = gemini_output_raw.strip() # Start with stripped raw output
                                    # Define common intro patterns (lowercase)
                                    common_intros = [
                                        "okay, here is", "sure, here is", "here is the provided text", "here's the refined text",
                                        "okay, here's the restructured text", "here is the text transformed", "here is the summary",
                                        "here is the documentation", "here is the narrative", "here are the questions"
                                        # Add more variations if you see them consistently
                                    ]
                                    # Try to find the first '## ' heading
                                    first_heading_pos = gemini_output_processed.find("## ")
                                    # Only attempt stripping if heading exists and isn't right at the start
                                    if first_heading_pos > 5: # Allow for a few spaces/newlines before heading
                                        # Extract the text before the first heading
                                        potential_intro_text = gemini_output_processed[:first_heading_pos].strip().lower()
                                        if potential_intro_text: # Only check if there's actual text before heading
                                            intro_found_and_stripped = False
                                            for intro_pattern in common_intros:
                                                if potential_intro_text.startswith(intro_pattern):
                                                    logger.warning(f"Detected and stripping potential intro phrase '{potential_intro_text[:50]}...' before first '##' heading in video {video_index+1}.")
                                                    # Overwrite with the text starting from the heading
                                                    gemini_output_processed = gemini_output_processed[first_heading_pos:]
                                                    self.status_update.emit(f"<font color='#f39c12'>Note: Auto-removed suspected intro text before first heading (Video {video_index+1}).</font>")
                                                    intro_found_and_stripped = True
                                                    break # Stop after stripping the first match
                                    # --- End Optional Post-processing ---

                                    # Write the potentially processed output
                                    final_f.write(gemini_output_processed.strip() + "\n") # Ensure final strip and newline
                                    processed_video_count += 1
                                    logger.info(f"Successfully processed and appended video {video_index + 1} to {self.output_file}")

                                except IOError as read_err:
                                    logger.error(f"Could not read temp file '{temp_response_file_path}' for video {video_index+1} after processing: {read_err}")
                                    final_f.write("**[Error reading processed content from temp file]**\n") # Add error marker
                            else:
                                # Video processing failed or was cancelled
                                final_f.write("**[Processing FAILED or CANCELLED for this video]**\n") # Add failure marker
                                logger.warning(f"Video {video_index+1} ({video_url}) processing failed or was cancelled; marker added to output.")

                            # Add a clear separator between videos using comments and Markdown rule
                            final_f.write(f"\n<!-- Processed Video End: {video_index + 1} -->\n\n---\n\n")
                        # --- <<< END REPLACEMENT BLOCK >>> ---
                    except IOError as e:
                        err_msg = f"Error appending to final output file '{self.output_file}' for video {video_index + 1}: {e}"
                        self.status_update.emit(f"<font color='{COLOR_ACCENT2}'>{err_msg}</font>")
                        logger.error(err_msg); self.error_occurred.emit(f"Fatal error writing final output: {e}"); break

                # Update overall progress
                progress_percent = int(((video_index + 1) / total_videos) * 100)
                self.progress_update.emit(progress_percent)

                if stop_video_processing: break # Exit video loop
            # --- End video loop ---

            # Final status after loop completion or break
            # ... (Final status logic remains the same) ...
            if not self._is_running:
                 final_msg = f"Gemini processing CANCELLED. Processed {processed_video_count}/{total_videos} videos before stopping."
                 status_color = '#e67e22'; self.status_update.emit(f"<font color='{status_color}'>{final_msg}</font>"); logger.info(final_msg)
            elif processed_video_count > 0:
                 all_succeeded = processed_video_count == total_videos
                 final_msg = f"Gemini processing finished. Successfully processed {processed_video_count}/{total_videos} videos."
                 status_color = COLOR_ACCENT1 if all_succeeded else '#f39c12'
                 self.status_update.emit(f"<font color='{status_color}'>{final_msg}</font>"); logger.info(final_msg)
                 self.processing_complete.emit(self.output_file)
            elif total_videos > 0:
                 err_msg = "Gemini processing finished, but failed to process any videos successfully."
                 self.error_occurred.emit(err_msg); logger.warning(err_msg)
            else:
                 err_msg = "Gemini processing finished, but no videos were provided from intermediate file."
                 self.error_occurred.emit(err_msg); logger.warning(err_msg)


        except Exception as e:
             # Catch any other unexpected errors in the main processing logic
             err_msg = f"Unexpected error during Gemini processing: {e}"
             self.error_occurred.emit(err_msg); logger.exception("Unhandled exception in GeminiProcessingThread run method:")
        finally:
            # Clean up the temporary file if it exists
            # ... (Temp file cleanup logic remains the same) ...
            if os.path.exists(temp_response_file_path):
                try:
                    os.remove(temp_response_file_path)
                    logger.debug(f"Removed temporary response file: {temp_response_file_path}")
                except OSError as e:
                    logger.warning(f"Could not remove temporary response file {temp_response_file_path}: {e}")
            logger.debug("GeminiProcessingThread run() method finished.")
            self.finished_signal.emit() # Always emit finished signal

    def stop(self):
        """Requests the thread to stop processing."""
        self.status_update.emit("Cancellation requested...")
        logger.info("Stop requested for GeminiProcessingThread.")
        self._is_running = False


# --- Main GUI Window ---

class MainWindow(QMainWindow):
    """Main application window for the YouTube Transcript & Gemini Refinement Tool."""

    AVAILABLE_MODELS: List[str] = [
        "gemini-1.5-flash-latest", "gemini-1.5-pro-latest",
        "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro",
        # Add experimental models here if needed and available via your API key
        "gemini-2.5-pro-exp-03-25",
        "gemini-2.0-flash-thinking-exp-01-21"
    ]

    def __init__(self):
        super().__init__()
        self.prompts_config, self.default_output_directory = load_config(DEFAULT_CONFIG_FILE)

        # Fallback for default output directory if not loaded or invalid
        if not self.default_output_directory or not os.path.isdir(self.default_output_directory):
             if self.default_output_directory: # Log if it was loaded but invalid
                 logger.warning(f"Loaded default output directory '{self.default_output_directory}' is invalid. Falling back to home dir.")
             else: # Log if it wasn't found in config
                 logger.info(f"No default output directory in config. Falling back to home dir.")
             self.default_output_directory = DEFAULT_OUTPUT_DIR_FALLBACK

        # Ensure prompts config is loaded
        if not self.prompts_config:
             logger.critical("Default prompts configuration is missing or invalid after load!")
             QMessageBox.critical(self, "Fatal Config Error", "Default prompt configuration is missing or invalid. Please check config.json or defaults.")
             # Provide minimal config to prevent crashes, though functionality will be limited
             self.prompts_config = {"Error": {"prompt": "Config Error", "chunk_size": DEFAULT_CHUNK_SIZE}}

        # Initialize state variables
        self.available_categories = list(self.prompts_config.keys())
        self.selected_category: str = DEFAULT_CATEGORY_NAME if DEFAULT_CATEGORY_NAME in self.available_categories else (self.available_categories[0] if self.available_categories else "No Categories")

        self.selected_model_name: str = DEFAULT_MODEL if DEFAULT_MODEL in self.AVAILABLE_MODELS else (self.AVAILABLE_MODELS[0] if self.AVAILABLE_MODELS else "No Models")
        if self.selected_model_name == "No Models":
             logger.error("No Gemini models available in AVAILABLE_MODELS list!")
        elif DEFAULT_MODEL not in self.AVAILABLE_MODELS and self.AVAILABLE_MODELS:
             logger.warning(f"Default model '{DEFAULT_MODEL}' not found. Using '{self.selected_model_name}' instead.")

        self.current_chunk_size: int = self._get_chunk_size_for_category(self.selected_category)
        self.preferred_transcript_languages: List[str] = DEFAULT_TRANSCRIPT_LANGUAGES
        self.extraction_thread: Optional[TranscriptExtractionThread] = None
        self.gemini_thread: Optional[GeminiProcessingThread] = None
        self.is_processing: bool = False
        self.current_intermediate_file_path: Optional[str] = None # Path to the temp transcript file

        self.initUI()
        self.apply_styling()
        self.load_initial_settings()
        logger.info("MainWindow initialized.")

    def _get_chunk_size_for_category(self, category_name: str) -> int:
        """Gets the suggested chunk size for the selected prompt category."""
        return self.prompts_config.get(category_name, {}).get('chunk_size', DEFAULT_CHUNK_SIZE)

    def _get_prompt_for_category(self, category_name: str) -> str:
        """Gets the prompt text for the selected category."""
        return self.prompts_config.get(category_name, {}).get('prompt', "Error: Prompt not found for category.")

    def load_initial_settings(self):
        """Loads initial values from environment or defaults into UI fields."""
        self.api_key_input.setText(os.environ.get("GEMINI_API_KEY", ""))
        self.language_input.setText(os.environ.get("DEFAULT_LANGUAGE", "English"))
        # Load transcript language preferences
        # You could use an environment variable like 'DEFAULT_TRANSCRIPT_LANGS'
        # Example: os.environ.get("DEFAULT_TRANSCRIPT_LANGS", "tr, en")
        default_transcript_langs_str = ", ".join(self.preferred_transcript_languages) # Use the initialized default
        self.transcript_lang_input.setText(default_transcript_langs_str)

        # Populate and select category (style)
        self.category_combo.blockSignals(True) # Avoid triggering change event on load
        self.category_combo.clear()
        if self.available_categories:
            self.category_combo.addItems(self.available_categories)
            if self.selected_category in self.available_categories:
                self.category_combo.setCurrentText(self.selected_category)
            elif self.available_categories: # Fallback if default wasn't valid
                self.category_combo.setCurrentIndex(0)
                self.selected_category = self.available_categories[0]
            self.category_combo.setEnabled(True)
        else:
            self.category_combo.addItem("No Styles Loaded")
            self.category_combo.setEnabled(False)
        self.category_combo.blockSignals(False)

        # Populate and select model
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        if self.AVAILABLE_MODELS:
            self.model_combo.addItems(self.AVAILABLE_MODELS)
            if self.selected_model_name in self.AVAILABLE_MODELS:
                self.model_combo.setCurrentText(self.selected_model_name)
            elif self.AVAILABLE_MODELS: # Fallback if default wasn't valid
                 self.model_combo.setCurrentIndex(0)
                 self.selected_model_name = self.AVAILABLE_MODELS[0]
            self.model_combo.setEnabled(True)
        else:
            self.model_combo.addItem("No Models Found")
            self.model_combo.setEnabled(False)
        self.model_combo.blockSignals(False)

        # Set chunk size slider
        self.chunk_size_slider.setMaximum(max(MAX_CHUNK_SIZE, self.current_chunk_size + 5000)) # Ensure max is reachable
        self.chunk_size_slider.setMinimum(MIN_CHUNK_SIZE)
        clamped_chunk_size = max(MIN_CHUNK_SIZE, min(self.current_chunk_size, MAX_CHUNK_SIZE))
        self.chunk_size_slider.setValue(clamped_chunk_size)
        self.update_chunk_size_label(clamped_chunk_size) # Update label without signal side effects

        # Display default output directory
        self.default_output_dir_input.setText(self.default_output_directory or "")
        self.default_output_dir_input.setCursorPosition(0)

        logger.debug("Initial settings loaded into UI.")


    # --- UI Initialization ---
    def initUI(self):
        """Initializes the main window UI elements and layout."""
        self.setWindowTitle("YouTube Transcript & Gemini Refinement Tool v4") # Bump version
        self.setGeometry(100, 100, 900, 850)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        main_layout.addWidget(self._create_title_section())
        main_layout.addWidget(self._create_input_container())
        main_layout.addWidget(self._create_progress_container())
        main_layout.addLayout(self._create_controls_section())

    def _create_title_section(self) -> QLabel:
        """Creates the main title label."""
        title_label = QLabel("YouTube Transcript & Gemini Refinement Tool")
        title_label.setFont(QFont(FONT_FAMILY, 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setObjectName("TitleLabel")
        return title_label

    def _create_input_container(self) -> QWidget:
        """Creates the container widget for all input fields."""
        input_container = QWidget()
        input_container.setObjectName("InputContainer")
        input_layout = QVBoxLayout(input_container)
        input_layout.setSpacing(10)

        # Row 1: URL Input
        url_layout = QVBoxLayout()
        url_layout.addWidget(self._create_styled_label("YouTube URL (Playlist or Video):"))
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter YouTube playlist or single video URL")
        url_layout.addWidget(self.url_input)
        input_layout.addLayout(url_layout)

        # Row 2: Language & API Key
        row2_layout = QHBoxLayout()
        lang_layout = QVBoxLayout()
        lang_layout.addWidget(self._create_styled_label("Output Language:"))
        self.language_input = QLineEdit()
        self.language_input.setPlaceholderText("e.g., English, Spanish")
        self.language_input.setToolTip("The language Gemini should primarily use for its output.")
        lang_layout.addWidget(self.language_input)
        row2_layout.addLayout(lang_layout)
        t_lang_layout = QVBoxLayout()
        t_lang_layout.addWidget(self._create_styled_label("Transcript Language(s) (Codes):"))
        self.transcript_lang_input = QLineEdit()
        self.transcript_lang_input.setPlaceholderText("e.g., tr, en, fr") # Shorter placeholder
        self.transcript_lang_input.setToolTip("Preferred transcript language codes (comma-separated, e.g., 'tr' for Turkish, 'en'). Will attempt fallback.")
        t_lang_layout.addWidget(self.transcript_lang_input)
        row2_layout.addLayout(t_lang_layout) # Add it to the same row

        api_layout = QVBoxLayout()
        api_layout.addWidget(self._create_styled_label("Gemini API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Enter Google AI Studio API key (or set GEMINI_API_KEY in .env)")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        api_layout.addWidget(self.api_key_input)
        row2_layout.addLayout(api_layout)
        input_layout.addLayout(row2_layout)

        # Row 3: Refinement Style & Model Selection
        row3_layout = QHBoxLayout()
        cat_layout = QVBoxLayout()
        cat_layout.addWidget(self._create_styled_label("Refinement Style:"))
        self.category_combo = QComboBox()
        # Items added in load_initial_settings
        self.category_combo.setToolTip("Select processing instructions (prompt) from config.json")
        self.category_combo.currentIndexChanged.connect(self._on_category_changed)
        cat_layout.addWidget(self.category_combo)
        row3_layout.addLayout(cat_layout)

        model_layout = QVBoxLayout()
        model_layout.addWidget(self._create_styled_label("Gemini Model:"))
        self.model_combo = QComboBox()
        # Items added in load_initial_settings
        self.model_combo.setToolTip("Select the Gemini model version to use")
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_layout.addWidget(self.model_combo)
        row3_layout.addLayout(model_layout)
        input_layout.addLayout(row3_layout)

        # Row 4: Chunk Size Slider
        input_layout.addLayout(self._create_chunk_size_section())

        # Row 5: Default Output Directory Selection
        input_layout.addWidget(self._create_styled_label("Default Final Output Directory:"))
        dir_layout = QHBoxLayout()
        dir_layout.setSpacing(6)
        self.default_output_dir_input = QLineEdit()
        self.default_output_dir_input.setReadOnly(True) # Display only
        self.default_output_dir_input.setPlaceholderText("Set default save location for final files...")
        self.default_output_dir_input.setToolTip("Default directory where the final output save dialog will start. Click 'Set Default' to change.")
        dir_layout.addWidget(self.default_output_dir_input, 1) # Stretch field
        self.set_default_dir_button = QPushButton("Set Default Dir...")
        self.set_default_dir_button.setObjectName("BrowseButton")
        self.set_default_dir_button.setToolTip("Choose and save the default directory for final outputs.")
        self.set_default_dir_button.clicked.connect(self.select_default_output_directory)
        dir_layout.addWidget(self.set_default_dir_button, 0) # No stretch
        input_layout.addLayout(dir_layout)

        # Row 6: Final Output File Path Selection/Confirmation
        self.gemini_file_input = self._create_file_input(
            input_layout,
            "Final Output File (.md - Auto-suggested):",
            "Select/Confirm Path...",
            self.select_gemini_output_file
        )
        self.gemini_file_input.setPlaceholderText("Path will be auto-suggested. Click button to override.")
        self.gemini_file_input.setToolTip("Path to save the final Gemini output. An automated name will be suggested based on title/style/time.")
        self.gemini_file_input.setObjectName("GeminiFileInput") # Set object name for styling/reference if needed

        return input_container

    def _create_chunk_size_section(self) -> QVBoxLayout:
        """Creates the chunk size slider and label group."""
        layout = QVBoxLayout()
        layout.setSpacing(5)
        label = self._create_styled_label("Gemini Request Chunk Size (Words - Approx):")
        layout.addWidget(label)

        slider_layout = QHBoxLayout()
        self.chunk_size_slider = QSlider(Qt.Horizontal)
        self.chunk_size_slider.setMinimum(MIN_CHUNK_SIZE)
        self.chunk_size_slider.setMaximum(MAX_CHUNK_SIZE) # Will be adjusted in load_initial_settings
        self.chunk_size_slider.setTickInterval(5000)
        self.chunk_size_slider.setTickPosition(QSlider.TicksBelow)
        self.chunk_size_slider.valueChanged.connect(self.update_chunk_size_label)
        self.chunk_size_slider.setToolTip("Adjusts the approximate number of words sent in each request to the Gemini API.")
        slider_layout.addWidget(self.chunk_size_slider)

        self.chunk_size_value_label = QLabel("...") # Initial text, updated on load
        self.chunk_size_value_label.setMinimumWidth(60)
        self.chunk_size_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        slider_layout.addWidget(self.chunk_size_value_label)
        layout.addLayout(slider_layout)

        desc = QLabel("(Affects tokens per request. Larger values use fewer API calls but may hit limits.)")
        desc.setObjectName("DescriptionLabel")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        return layout

    def _create_file_input(self, parent_layout: QVBoxLayout, label_text: str, button_text: str, handler) -> QLineEdit:
        """Helper to create a read-only line edit with a browse button."""
        parent_layout.addWidget(self._create_styled_label(label_text))
        layout = QHBoxLayout()
        layout.setSpacing(6)
        input_field = QLineEdit()
        input_field.setReadOnly(True) # Path is selected via dialog
        input_field.setPlaceholderText(f"Click '{button_text}' to select path")
        layout.addWidget(input_field, 1) # Field stretches
        button = QPushButton(button_text)
        button.setObjectName("BrowseButton") # Style as a browse button
        button.clicked.connect(handler)
        layout.addWidget(button, 0) # Button doesn't stretch
        parent_layout.addLayout(layout)
        return input_field

    def _create_progress_container(self) -> QWidget:
        """Creates the container for the progress bar and status display."""
        progress_container = QWidget()
        progress_container.setObjectName("ProgressContainer")
        progress_layout = QVBoxLayout(progress_container)
        progress_layout.setSpacing(10)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        progress_layout.addWidget(self.progress_bar)

        self.status_display = QTextEdit()
        self.status_display.setReadOnly(True)
        self.status_display.setObjectName("StatusDisplay")
        self.status_display.setPlaceholderText("Status messages will appear here...")
        self.status_display.setMinimumHeight(150) # Ensure decent space for messages
        progress_layout.addWidget(self.status_display)
        return progress_container

    def _create_controls_section(self) -> QHBoxLayout:
        """Creates the layout for the Start and Cancel buttons."""
        control_layout = QHBoxLayout()
        control_layout.setSpacing(20)
        self.extract_button = QPushButton("Start Processing")
        self.extract_button.setObjectName("StartButton") # For specific styling
        self.extract_button.setToolTip("Begin extracting transcripts and processing with Gemini.")
        self.extract_button.clicked.connect(self.start_processing)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("CancelButton") # For specific styling
        self.cancel_button.setToolTip("Stop the current processing task.")
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.cancel_button.setEnabled(False) # Disabled initially

        control_layout.addStretch(1) # Center buttons
        control_layout.addWidget(self.extract_button)
        control_layout.addWidget(self.cancel_button)
        control_layout.addStretch(1)
        return control_layout

    def _create_styled_label(self, text: str) -> QLabel:
        """Helper to create a standard bold label."""
        label = QLabel(text)
        label.setFont(QFont(FONT_FAMILY, 10, QFont.Bold))
        return label

    # --- Styling ---
    def apply_styling(self):
        """Applies the dark theme palette and CSS stylesheet."""
        self.set_dark_palette()
        # Adjusted BrowseButton min-width for longer text
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {COLOR_BACKGROUND}; }}
            QWidget {{ color: {COLOR_TEXT}; font-family: {FONT_FAMILY}; font-size: 10pt; }}
            #InputContainer, #ProgressContainer {{
                background-color: {COLOR_WIDGET_BACKGROUND};
                border-radius: 8px;
                padding: 18px;
                border: 1px solid {COLOR_BORDER};
                margin-bottom: 10px;
            }}
            QLabel {{ color: {COLOR_TEXT}; padding-bottom: 2px; }}
            #TitleLabel {{
                color: {COLOR_ACCENT1}; font-size: 18pt; font-weight: bold;
                padding: 15px; margin-bottom: 15px; border-radius: 5px;
                background-color: {COLOR_WIDGET_BACKGROUND}; border: 1px solid {COLOR_ACCENT1};
            }}
            #DescriptionLabel {{
                color: {COLOR_PLACEHOLDER}; font-size: 8pt; font-style: italic; padding-top: 5px;
            }}
            QLineEdit {{
                background-color: {COLOR_BACKGROUND}; border: 1px solid {COLOR_BORDER};
                border-radius: 4px; color: {COLOR_TEXT}; padding: 7px 10px;
                font-size: 9pt; min-height: 20px;
            }}
            QLineEdit:read-only {{
                background-color: {COLOR_WIDGET_BACKGROUND}; color: {COLOR_PLACEHOLDER};
                border: 1px solid {COLOR_PLACEHOLDER}; font-style: italic;
            }}
            QLineEdit:disabled {{
                background-color: {COLOR_WIDGET_BACKGROUND}; border-color: {COLOR_PLACEHOLDER};
                color: {COLOR_PLACEHOLDER};
            }}
            QComboBox {{
                background-color: {COLOR_BACKGROUND}; border: 1px solid {COLOR_BORDER};
                border-radius: 4px; padding: 7px 10px; min-width: 6em; min-height: 20px;
            }}
            QComboBox:disabled {{
                background-color: {COLOR_WIDGET_BACKGROUND}; border-color: {COLOR_PLACEHOLDER};
                color: {COLOR_PLACEHOLDER};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding; subcontrol-position: top right; width: 22px;
                border-left-width: 1px; border-left-color: {COLOR_BORDER}; border-left-style: solid;
                border-top-right-radius: 3px; border-bottom-right-radius: 3px;
            }}
            QComboBox::down-arrow {{ image: url(down_arrow.png); /* Needs an actual image or use default */ width: 12px; height: 12px; }}
            QComboBox QAbstractItemView {{
                border: 1px solid {COLOR_BORDER}; background-color: {COLOR_WIDGET_BACKGROUND};
                color: {COLOR_TEXT}; selection-background-color: {COLOR_BORDER};
                selection-color: {COLOR_BACKGROUND}; padding: 5px; outline: 0px;
            }}
            QPushButton {{
                background-color: {COLOR_BORDER}; color: white; border: none;
                border-radius: 4px; padding: 9px 20px; font-size: 10pt; font-weight: bold;
                min-width: 90px; outline: none;
            }}
            QPushButton:hover {{ background-color: #4aa3df; }}
            QPushButton:pressed {{ background-color: #2a8cd0; }}
            QPushButton:disabled {{ background-color: {COLOR_PLACEHOLDER}; color: {COLOR_BACKGROUND}; }}
            #StartButton {{ background-color: {COLOR_ACCENT1}; }}
            #StartButton:hover {{ background-color: #29b765; }}
            #StartButton:pressed {{ background-color: #24a159; }}
            #CancelButton {{ background-color: {COLOR_ACCENT2}; }}
            #CancelButton:hover {{ background-color: #e4604f; }}
            #CancelButton:pressed {{ background-color: #d63a27; }}
            #BrowseButton {{ /* Style for file/dir selection buttons */
                padding: 7px 12px; font-size: 9pt; font-weight: normal; min-width: 110px; /* Wider for new text */
                background-color: {COLOR_ACCENT3}; color: {COLOR_BACKGROUND};
            }}
            #BrowseButton:hover {{ background-color: #aab7b8; }}
            #BrowseButton:pressed {{ background-color: #849394; }}
            QProgressBar {{
                border: 1px solid {COLOR_BORDER}; border-radius: 4px; text-align: center;
                color: {COLOR_TEXT}; font-size: 9pt; height: 24px;
                background-color: {COLOR_BACKGROUND};
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {COLOR_BORDER}, stop:1 {COLOR_ACCENT1});
                border-radius: 3px; margin: 1px;
            }}
            #StatusDisplay {{
                background-color: {COLOR_BACKGROUND}; border: 1px solid {COLOR_BORDER};
                border-radius: 4px; color: {COLOR_TEXT};
                font-family: Consolas, Courier New, monospace; font-size: 9pt; padding: 10px;
            }}
            QSlider::groove:horizontal {{
                border: 1px solid {COLOR_PLACEHOLDER}; height: 4px; background: {COLOR_BACKGROUND};
                margin: 2px 0; border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {COLOR_BORDER}; border: 1px solid {COLOR_BORDER};
                width: 18px; margin: -8px 0; /* Vertically center handle */
                border-radius: 9px;
            }}
            QSlider::handle:horizontal:hover {{ background: #4aa3df; border: 1px solid #4aa3df; }}
            QToolTip {{
                background-color: {COLOR_BACKGROUND}; color: {COLOR_TEXT};
                border: 1px solid {COLOR_BORDER}; padding: 6px; opacity: 230; border-radius: 3px;
            }}
        """)

    def set_dark_palette(self):
        """Sets a dark color palette for the application."""
        QApplication.setStyle(QStyleFactory.create('Fusion'))
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(COLOR_BACKGROUND))
        dark_palette.setColor(QPalette.WindowText, QColor(COLOR_TEXT))
        dark_palette.setColor(QPalette.Base, QColor(COLOR_WIDGET_BACKGROUND))
        dark_palette.setColor(QPalette.AlternateBase, QColor(COLOR_BACKGROUND))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(COLOR_WIDGET_BACKGROUND))
        dark_palette.setColor(QPalette.ToolTipText, QColor(COLOR_TEXT))
        dark_palette.setColor(QPalette.Text, QColor(COLOR_TEXT))
        dark_palette.setColor(QPalette.Button, QColor(COLOR_WIDGET_BACKGROUND))
        dark_palette.setColor(QPalette.ButtonText, QColor(COLOR_TEXT))
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(COLOR_BORDER))
        dark_palette.setColor(QPalette.Highlight, QColor(COLOR_BORDER))
        dark_palette.setColor(QPalette.HighlightedText, QColor(COLOR_BACKGROUND))

        # Disabled state colors
        disabled_text_color = QColor(COLOR_PLACEHOLDER)
        disabled_widget_bg = QColor(COLOR_WIDGET_BACKGROUND).darker(110) # Slightly darker disabled bg
        dark_palette.setColor(QPalette.Disabled, QPalette.WindowText, disabled_text_color)
        dark_palette.setColor(QPalette.Disabled, QPalette.Text, disabled_text_color)
        dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, disabled_text_color)
        dark_palette.setColor(QPalette.Disabled, QPalette.Base, disabled_widget_bg) # Background for disabled input widgets
        dark_palette.setColor(QPalette.Disabled, QPalette.Button, disabled_widget_bg) # Background for disabled buttons
        dark_palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(80, 80, 80))
        dark_palette.setColor(QPalette.Disabled, QPalette.HighlightedText, disabled_text_color)

        QApplication.setPalette(dark_palette)

    # --- Event Handlers & Slots ---
    @pyqtSlot(int)
    def _on_category_changed(self, index: int):
        """Handles selection changes in the refinement style combo box."""
        if index == -1 or not self.available_categories: return
        category_name = self.category_combo.itemText(index)
        if category_name not in self.prompts_config:
            logger.warning(f"Selected category '{category_name}' not found in config.")
            return

        self.selected_category = category_name
        suggested_chunk_size = self._get_chunk_size_for_category(category_name)
        logger.info(f"Category changed to: '{category_name}' (Suggested chunk size: {suggested_chunk_size})")

        # Update slider without triggering its valueChanged signal during adjustment
        self.chunk_size_slider.blockSignals(True)
        clamped_chunk_size = max(MIN_CHUNK_SIZE, min(suggested_chunk_size, MAX_CHUNK_SIZE))
        self.chunk_size_slider.setValue(clamped_chunk_size)
        self.chunk_size_slider.blockSignals(False)

        # Update internal state and label directly
        self.current_chunk_size = clamped_chunk_size
        self.update_chunk_size_label(clamped_chunk_size)
        self.update_status(f"Style set to '{category_name}'. Chunk size updated to {clamped_chunk_size}.")

    @pyqtSlot(str)
    def _on_model_changed(self, model_name: str):
        """Handles selection changes in the Gemini model combo box."""
        if model_name in self.AVAILABLE_MODELS:
            self.selected_model_name = model_name
            logger.info(f"Gemini model changed to: {model_name}")
            self.update_status(f"Gemini model set to '{model_name}'.")
        elif model_name: # Log if a non-empty, invalid name appears somehow
            logger.warning(f"Invalid or unknown model selected: {model_name}")

    @pyqtSlot(int)
    def update_chunk_size_label(self, value: int):
        """Updates the chunk size label and internal variable when the slider changes."""
        self.chunk_size_value_label.setText(str(value))
        self.current_chunk_size = value
        # Optional: log this change if slider moved by user
        # if self.chunk_size_slider.signalsBlocked() is False:
        #    logger.debug(f"Chunk size slider changed to: {value}")

    @pyqtSlot(int)
    def _update_extraction_progress(self, value: int):
        """Updates the progress bar during transcript extraction."""
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"Extracting: {value}%")

    @pyqtSlot(int)
    def _update_gemini_progress(self, value: int):
        """Updates the progress bar during Gemini processing."""
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"Processing: {value}%")

    @pyqtSlot(str)
    def update_status(self, message: str):
        """Appends a message to the status display box, applying color coding."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        # Basic HTML check
        is_html = "<" in message and ">" in message and ("<font" in message or "<br" in message)

        if is_html:
            # Assume pre-formatted HTML message includes timestamp if needed
            self.status_display.append(message)
        else:
            # Apply color coding based on keywords
            color = COLOR_TEXT # Default color
            lower_message = message.lower()
            if any(err in lower_message for err in ["error", "failed", "invalid", "denied", "blocked", "fatal", "critical", "exception"]):
                color = COLOR_ACCENT2 # Red
            elif any(warn in lower_message for warn in ["warning", "skipped", "unavailable", "disabled", "empty response", "failed to get", "could not delete", "partial"]):
                color = "#f39c12" # Orange/Yellow
            elif any(ok in lower_message for ok in ["complete", "success", "finished", "processed", "saved to", "extraction complete", "starting gemini", "deleted intermediate", "found playlist", "processing video"]):
                color = COLOR_ACCENT1 # Green
            elif "cancel" in lower_message:
                color = "#e67e22" # Darker Orange
            # Append with timestamp and color
            self.status_display.append(f"<font color='{color}'>[{timestamp}] {message}</font>")

        # Auto-scroll to the bottom
        self.status_display.verticalScrollBar().setValue(self.status_display.verticalScrollBar().maximum())
        QApplication.processEvents() # Ensure UI updates during processing

    @pyqtSlot(str)
    def _on_extraction_complete(self, intermediate_transcript_file: str):
        """Handles the completion of the transcript extraction thread."""
        self.progress_bar.setValue(0) # Reset for Gemini phase
        self.progress_bar.setFormat("Processing: 0%")
        self.update_status(f"<font color='{COLOR_ACCENT1}'>Transcript extraction complete!</font> Intermediate file: {os.path.basename(intermediate_transcript_file)}")
        self.update_status("Starting Gemini processing...")
        logger.info(f"Extraction complete (Intermediate: {intermediate_transcript_file}). Starting Gemini processing.")

        selected_prompt_text = self._get_prompt_for_category(self.selected_category)
        current_chunk_value = self.chunk_size_slider.value() # Use current slider value
        final_output_md_path = self.gemini_file_input.text().strip()
        api_key = self.api_key_input.text().strip()
        output_lang = self.language_input.text().strip() or "English" # Default to English if empty

        # Basic validation before starting Gemini thread
        if not final_output_md_path or not final_output_md_path.lower().endswith(".md"):
            self._handle_error("Invalid or missing final Gemini output file path (.md required).")
            self.set_processing_state(False); return
        if not os.path.exists(intermediate_transcript_file):
             self._handle_error(f"Intermediate transcript file missing: {intermediate_transcript_file}")
             self.set_processing_state(False); return
        if not api_key:
            self._handle_error("Gemini API Key is missing.")
            self.set_processing_state(False); return

        # Create and start the Gemini processing thread
        self.gemini_thread = GeminiProcessingThread(
            intermediate_transcript_file=intermediate_transcript_file,
            final_output_file_md=final_output_md_path,
            api_key=api_key,
            model_name=self.selected_model_name,
            output_language=output_lang,
            chunk_size=current_chunk_value,
            prompt_template=selected_prompt_text,
            style_category=self.selected_category, # Pass the style name
            parent=self
        )
        # Connect signals
        self.gemini_thread.progress_update.connect(self._update_gemini_progress)
        self.gemini_thread.status_update.connect(self.update_status)
        self.gemini_thread.processing_complete.connect(self._on_processing_success)
        self.gemini_thread.error_occurred.connect(self._handle_error)
        self.gemini_thread.finished_signal.connect(self._on_gemini_thread_finished)
        self.gemini_thread.finished.connect(self.gemini_thread.deleteLater) # Qt cleanup
        self.gemini_thread.start()


    @pyqtSlot(str)
    def _on_processing_success(self, original_output_path: str):
        """
        Handles successful completion of the entire process.
        Attempts to rename the output file based on the first H2 title '## **...**' found.
        """
        logger.info(f"Processing completed successfully. Initial output path: {original_output_path}")
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Complete!")
        logger.debug(f"RENAME_DEBUG: Starting _on_processing_success for: {original_output_path}") # ADDED

        final_path_to_show = original_output_path # Default to original path

        # --- Attempt to rename file based on content ---
        try:
            # Call the updated title extractor
            # Ensure _extract_title_from_file uses the new comment marker internally
            logger.debug("RENAME_DEBUG: Calling _extract_title_from_file...") # ADDED
            extracted_title = self._extract_title_from_file(original_output_path) # <<<< Uses the function from point 1
            logger.debug(f"RENAME_DEBUG: _extract_title_from_file returned: '{extracted_title}'") # ADDED

            if extracted_title: # <<<< Check 1: Was a title actually found?
                logger.debug(f"RENAME_DEBUG: Calling _generate_renamed_path with title: '{extracted_title}'") # ADDED
                logger.info(f"Extracted title for renaming: '{extracted_title}'")
                # Generate the new path based on the extracted title
                new_path = self._generate_renamed_path(original_output_path, extracted_title) # <<<< Uses helper to sanitize etc.
                logger.debug(f"RENAME_DEBUG: _generate_renamed_path returned: '{new_path}'") # ADDED

                if new_path and new_path != original_output_path: # <<<< Check 2: Is new path valid and different?
                    logger.debug(f"RENAME_DEBUG: New path is valid and different. Checking existence of '{new_path}'...") # ADDED
                    if not os.path.exists(new_path): # <<<< Check 3: Does the new file name already exist?
                        logger.debug(f"RENAME_DEBUG: Target path does not exist. Attempting os.rename...") # ADDED
                        try:
                            os.rename(original_output_path, new_path) # <<<< The actual rename
                            logger.info(f"RENAME_DEBUG: os.rename successful!") # ADDED log level info
                            self.update_status(f"<font color='{COLOR_ACCENT1}'>Renamed output file based on content to: {os.path.basename(new_path)}</font>")
                            final_path_to_show = new_path # Use the new path for messages
                        except OSError as rename_err:
                            logger.error(f"RENAME_DEBUG: os.rename FAILED: {rename_err}", exc_info=True) # ADDED exc_info
                            self.update_status(f"<font color='#f39c12'>Warning: Could not rename output file: {rename_err}</font>")
                    else:
                        # New file name already exists, don't overwrite
                        logger.warning(f"RENAME_DEBUG: Skipping rename because target file already exists: '{new_path}'") # ADDED
                        self.update_status(f"<font color='#f39c12'>Warning: Skipping rename, file '{os.path.basename(new_path)}' already exists.</font>")
                elif new_path == original_output_path:
                     logger.info("RENAME_DEBUG: Generated name is same as original. No rename needed.") # ADDED
                else:
                     logger.warning(f"RENAME_DEBUG: Generated new path was invalid or empty. No rename attempted.") # ADDED


            else: # <<<< This happens if _extract_title_from_file returned None
                logger.info("RENAME_DEBUG: No title extracted. File will not be renamed.") # ADDED
                # Optionally add a status update if you want user feedback on this
                # self.update_status("<font color='#f39c12'>Note: Could not find title in content for renaming.</font>")

        except Exception as e:
            # Catch any unexpected error during the rename attempt
            logger.error(f"RENAME_DEBUG: Exception during rename attempt block: {e}", exc_info=True) # ADDED exc_info
            self.update_status(f"<font color='#f39c12'>Warning: Error occurred during file rename attempt.</font>")
        # --- End rename attempt ---

        # Delete intermediate file (uses self.current_intermediate_file_path)
        self._delete_intermediate_file() # <<< This should happen regardless of rename success

        # Show final success message using the final path (original or renamed)
        self.update_status(f"<font color='{COLOR_ACCENT1}'>Processing complete! Output saved to:</font> {final_path_to_show}")
        QMessageBox.information(self, "Success",
                                f"Processing complete!\n\nOutput saved to:\n{final_path_to_show}",
                                QMessageBox.Ok)
        logger.debug(f"RENAME_DEBUG: Finished _on_processing_success. Final path to show: {final_path_to_show}") # ADDED

    # <<< NEW HELPER METHOD 1 >>>
    def _extract_title_from_file(self, file_path: str) -> Optional[str]:
        """
        Reads the file content after the YAML front matter and looks for
        the first H2 heading formatted as '## **Title Text**' after the
        '<!-- Processed Video Start: ... -->' marker, extracting 'Title Text'.
        """
        logger.debug(f"RENAME_DEBUG: Starting title extraction for: {file_path}")
        try:
            # Regex to find the specific title pattern '## **Title**'
            title_pattern = re.compile(r"^## \*\*(.+?)\*\*\s*$") # Pattern looks correct
            # Regex to find the start marker comment
            start_marker_pattern = re.compile(r"^<!-- Processed Video Start: \d+ -->\s*$") # Use the new comment marker

            logger.debug(f"Attempting to extract title from: {file_path} using pattern: {title_pattern.pattern}")

            with open(file_path, 'r', encoding='utf-8') as f:
                lines_to_check = 50 # Check a reasonable number of lines after the marker
                lines_read_after_marker = 0
                in_yaml = False
                found_video_marker = False # Flag to track if we've seen the video start comment
                line_counter = 0

                for line in f:
                    line_counter += 1
                    stripped_line = line.strip()
                    # Don't log every line read by default unless needed, can be very verbose
                    # logger.debug(f"RENAME_DEBUG: Reading line {line_counter}: '{stripped_line[:80]}...'")

                    # --- START: REVISED YAML Skipping Logic ---
                    # Check for start of YAML block ONLY on the first line
                    if line_counter == 1 and stripped_line == '---':
                        in_yaml = True
                        logger.debug("RENAME_DEBUG: Entered YAML block on line 1.")
                        continue # Skip the first '---'

                    # Check for end of YAML block if we are inside one
                    if in_yaml:
                        if stripped_line == '---':
                            in_yaml = False
                            logger.debug(f"RENAME_DEBUG: Exited YAML block on line {line_counter}.")
                            continue # Skip the closing '---'
                        else:
                            # Still inside YAML, skip this line entirely
                            continue
                    # --- END: REVISED YAML Skipping Logic ---

                    # 2. Look for the Video Start marker comment
                    if not found_video_marker:
                        if start_marker_pattern.match(stripped_line):
                            found_video_marker = True
                            logger.debug(f"RENAME_DEBUG: Found video start marker on line {line_counter}") # ADDED
                            continue # Move to the next line after finding the marker

                    # 3. Only look for the title pattern *after* finding the marker
                    if found_video_marker:
                        logger.debug(f"RENAME_DEBUG: Line {line_counter}: Checking for title pattern match.") # ADDED
                        match = title_pattern.match(stripped_line)
                        if match:
                            # Extract the text captured in group 1
                            title = match.group(1).strip()
                            logger.debug(f"RENAME_DEBUG: Line {line_counter}: Title pattern MATCHED. Extracted raw: '{match.group(1)}', Stripped: '{title}'") # ADDED
                            if title: # Ensure title is not empty
                                logger.info(f"RENAME_DEBUG: Found valid title: '{title}'. Returning.") # MODIFIED log level
                                return title # Return the first valid title found
                            else:
                                logger.warning(f"RENAME_DEBUG: Line {line_counter}: Pattern matched but extracted title was empty.") # ADDED
                        # else: logger.debug(f"RENAME_DEBUG: Line {line_counter}: No title pattern match.") # Optional: too verbose?

                        # Increment lines read counter *after* finding marker
                        lines_read_after_marker += 1
                        if lines_read_after_marker >= lines_to_check:
                            logger.debug(f"RENAME_DEBUG: Reached check limit ({lines_to_check}) after marker. Stopping search.") # ADDED
                            break # Stop after checking enough lines after the marker

        except IOError as e:
            logger.error(f"Could not read file '{file_path}' to extract title: {e}")
        except Exception as e:
             logger.error(f"RENAME_DEBUG: Exception during title extraction: {e}", exc_info=True) # ADDED exc_info

        logger.warning("RENAME_DEBUG: Title extraction finished without returning a title.") # ADDED
        return None # Title not found or error occurred


    # <<< NEW HELPER METHOD 2 >>>
    def _generate_renamed_path(self, original_path: str, title: str) -> Optional[str]:
        """
        Generates a new filename based on the title, removing invalid characters
        but keeping spaces, and omitting the timestamp.
        """
        try:
            # 1. Sanitize the extracted title - Keep spaces, remove truly invalid chars
            # Define characters invalid in Windows/Mac/Linux filenames (approximate common set)
            invalid_chars = r'[<>:"/\\|?*\x00-\x1F]' # Control chars + standard invalid ones
            sanitized_title = re.sub(invalid_chars, '', title)

            # Replace multiple spaces with single space, strip leading/trailing space
            sanitized_title = re.sub(r'\s+', ' ', sanitized_title).strip()

            # Optionally remove leading/trailing periods or spaces again after reduction
            sanitized_title = sanitized_title.strip('. ')

            # 2. Limit length (e.g., to 100 characters) - Adjust as needed
            max_len = 100
            if len(sanitized_title) > max_len:
                # Try to trim at the last space before the limit
                trim_pos = sanitized_title.rfind(' ', 0, max_len)
                if trim_pos > max_len / 2: # Only trim at space if it's reasonably far in
                    sanitized_title = sanitized_title[:trim_pos]
                else: # Otherwise, just cut at the max length
                    sanitized_title = sanitized_title[:max_len]
                # Strip again after trimming
                sanitized_title = sanitized_title.strip('. ')


            if not sanitized_title:
                logger.warning("Sanitized title is empty after removing invalid characters and limiting length, cannot rename.")
                return None

            # 3. Get the directory from the original path
            directory = os.path.dirname(original_path)
            if not directory: # Handle case where original path might just be a filename
                 directory = "."

            # 4. Construct the new filename WITHOUT the timestamp
            new_filename = f"{sanitized_title}.md"
            logger.debug(f"Generated new filename (no timestamp, spaces kept): {new_filename}")

            # 5. Return the full new path
            return os.path.join(directory, new_filename)

        except Exception as e:
            logger.error(f"Error generating renamed path from title '{title}': {e}", exc_info=True)
            return None


    def _delete_intermediate_file(self):
        """Attempts to delete the intermediate transcript file."""
        # --- Keep this method exactly as you have it ---
        intermediate_to_delete = self.current_intermediate_file_path
        if intermediate_to_delete and os.path.exists(intermediate_to_delete):
            try:
                os.remove(intermediate_to_delete)
                logger.info(f"Successfully deleted intermediate file: {intermediate_to_delete}")
                self.update_status(f"Deleted intermediate file: {os.path.basename(intermediate_to_delete)}")
            except OSError as e:
                logger.error(f"Could not delete intermediate file '{intermediate_to_delete}': {e}")
                self.update_status(f"<font color='#f39c12'>Warning: Could not delete intermediate file: {os.path.basename(intermediate_to_delete)} ({e})</font>")
        elif intermediate_to_delete:
            logger.warning(f"Intermediate file path recorded ('{intermediate_to_delete}') but file not found for deletion.")
        else:
            logger.debug("No intermediate file path recorded, skipping deletion.")
        self.current_intermediate_file_path = None


    @pyqtSlot(str)
    def _handle_error(self, error_message: str):
        """Handles errors reported by worker threads or validation."""
        # Ensure message starts with ERROR for consistency if not already formatted
        display_message = f"ERROR: {error_message}" if not error_message.lower().startswith(("error", "warning", "gemini api error", "invalid")) else error_message
        self.update_status(f"<font color='{COLOR_ACCENT2}'>{display_message}</font>")
        logger.error(f"Error handled: {error_message}") # Log the original message
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Error!")
        QApplication.processEvents() # Ensure UI updates

        # Check if threads are finished before resetting UI (avoid race condition)
        should_reset_ui = False
        if self.is_processing: # Only reset if we were in a processing state
            ext_thread_running = self.extraction_thread and self.extraction_thread.isRunning()
            gem_thread_running = self.gemini_thread and self.gemini_thread.isRunning()
            if not ext_thread_running and not gem_thread_running:
                 logger.warning("Error occurred, and both threads (if started) have finished. Resetting UI.")
                 should_reset_ui = True
            else:
                 logger.warning("Error occurred, but a processing thread might still be running (or stopping). UI state not reset yet.")

        # Show message box to the user
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Processing Error")
        msg_box.setText("An error occurred during processing:")
        # Allow selecting text in the informative part
        msg_box.setInformativeText(error_message)
        msg_box.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

        if should_reset_ui:
            self.set_processing_state(False)

        # Do NOT delete intermediate file on error, user might need it for debugging
        # Clear the path variable though, as the current run is over
        self.current_intermediate_file_path = None

    @pyqtSlot()
    def _on_extraction_thread_finished(self):
        """Handles the finished signal from the extraction thread."""
        logger.debug("TranscriptExtractionThread finished signal received.")
        self.extraction_thread = None # Clear reference
        # If Gemini thread hasn't started (e.g., extraction error/cancel), reset UI
        if not self.gemini_thread and self.is_processing:
            logger.debug("Extraction finished, no Gemini thread active. Resetting UI state.")
            self.set_processing_state(False)
            # Reset progress bar if it doesn't show Error or Complete
            current_format = self.progress_bar.format()
            if "Error!" not in current_format and "Complete!" not in current_format:
                self.progress_bar.setValue(0)
                self.progress_bar.setFormat("%p%")
            # Clear intermediate path if extraction failed/cancelled before Gemini started
            self.current_intermediate_file_path = None


    @pyqtSlot()
    def _on_gemini_thread_finished(self):
        """Handles the finished signal from the Gemini processing thread."""
        logger.debug("GeminiProcessingThread finished signal received.")
        self.gemini_thread = None # Clear reference
        if self.is_processing: # Reset UI if we were processing
            logger.debug("Gemini thread finished. Resetting processing state.")
            self.set_processing_state(False)
            # Reset progress bar only if it doesn't show Complete or Error
            current_format = self.progress_bar.format()
            if "Complete!" not in current_format and "Error!" not in current_format:
                 self.progress_bar.setValue(0)
                 self.progress_bar.setFormat("%p%")

        # Intermediate file deletion is handled ONLY in _on_processing_success.
        # Clear the variable here if it wasn't cleared by success/error, just in case.
        if self.current_intermediate_file_path:
            logger.debug("Gemini finished (status uncertain), ensuring intermediate path variable is cleared.")
            self.current_intermediate_file_path = None


    # --- Actions ---

    def validate_inputs(self) -> bool:
        """Validates user inputs before starting processing."""
        logger.debug("Validating inputs...")
        url = self.url_input.text().strip()
        lang = self.language_input.text().strip()
        api_key = self.api_key_input.text().strip()
        gemini_file_md = self.gemini_file_input.text().strip() # Path shown in UI
        category = self.category_combo.currentText()

        # Core field checks
        if not url: return self._show_validation_error("YouTube URL is required.")
        if not ("youtube.com/" in url or "youtu.be/" in url): return self._show_validation_error("Please enter a valid YouTube video or playlist URL.")
        if not lang: return self._show_validation_error("Output Language is required.")
        if not api_key: return self._show_validation_error("Gemini API Key is required (or set GEMINI_API_KEY in .env).")
        if not gemini_file_md: return self._show_validation_error("Final Gemini output file path is required (use 'Select/Confirm Path...').")
        if not gemini_file_md.lower().endswith(".md"): return self._show_validation_error("Final Gemini output file must have a .md extension.")
        if not self.available_categories or category not in self.prompts_config: return self._show_validation_error("A valid Refinement Style must be selected.")

        # Final output directory writability check
        final_dir_path = os.path.dirname(gemini_file_md)
        if not final_dir_path: final_dir_path = "." # Use current directory if only filename given

        if not os.path.exists(final_dir_path):
             if self._show_confirmation(f"The directory for the final output:\n'{final_dir_path}'\ndoes not exist. Create it now?"):
                 try: os.makedirs(final_dir_path, exist_ok=True); logger.info(f"Created missing directory: {final_dir_path}")
                 except OSError as e: return self._show_validation_error(f"Failed to create directory '{final_dir_path}': {e}")
             else: self.update_status("Directory creation cancelled by user."); return False # User said no
        elif not os.path.isdir(final_dir_path): return self._show_validation_error(f"The final output path's directory is not valid: '{final_dir_path}'.")
        elif not os.access(final_dir_path, os.W_OK): return self._show_validation_error(f"No permission to write to the final output directory: '{final_dir_path}'.")

        # Intermediate file directory (script dir or cwd) writability check
        try:
             script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
             logger.warning("__file__ not defined, checking current working directory for intermediate file write access.")
             script_dir = "." # Fallback to current working directory

        if not os.access(script_dir, os.W_OK):
             return self._show_validation_error(f"No permission to write intermediate file to directory: '{script_dir}'. Check permissions.")

        logger.info("Input validation successful.")
        return True

    def _show_validation_error(self, message: str) -> bool:
        """Displays a validation error message box."""
        logger.warning(f"Input validation failed: {message}")
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("Input Validation Error")
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
        return False # Indicates validation failed

    def _show_confirmation(self, message: str) -> bool:
        """Displays a confirmation dialog (Yes/No)."""
        logger.debug(f"Showing confirmation dialog: {message}")
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setWindowTitle("Confirmation Required")
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No) # Default to No
        return msg_box.exec_() == QMessageBox.Yes

    def set_processing_state(self, processing: bool):
        """Enables/disables UI elements based on processing state."""
        self.is_processing = processing
        self.extract_button.setEnabled(not processing)
        self.cancel_button.setEnabled(processing)

        # List of widgets to disable/enable
        widgets_to_toggle = [
            self.url_input, self.language_input, self.api_key_input,
            self.category_combo, self.chunk_size_slider, self.model_combo,
            self.set_default_dir_button, # Disable changing default dir while running
            self.gemini_file_input # Disable the field itself (though read-only)
        ]
        for widget in widgets_to_toggle:
            widget.setEnabled(not processing)

        # Also disable the button associated with the final output path QLineEdit
        field = self.gemini_file_input
        parent_widget = field.parentWidget() # Get the widget containing the layout
        if parent_widget:
            parent_layout = parent_widget.layout()
            if parent_layout and isinstance(parent_layout, QHBoxLayout):
                # Assuming button is the second item (index 1) in the QHBoxLayout
                if parent_layout.count() > 1:
                    button_item = parent_layout.itemAt(1)
                    if button_item and button_item.widget() and isinstance(button_item.widget(), QPushButton):
                        button_item.widget().setEnabled(not processing)

        logger.debug(f"Processing state set to: {processing}. UI elements updated.")


    def start_processing(self):
        """Starts the transcript extraction and Gemini processing workflow."""
        if self.is_processing:
            logger.warning("Start processing called while already processing. Ignoring.")
            return

        # Auto-populate final filename if empty before validation
        if not self.gemini_file_input.text().strip():
            logger.info("Final output path empty on start, attempting to auto-populate.")
            # Call _select_output_file in populate mode (doesn't show dialog)
            self._select_output_file("Auto-generating filename...", self.gemini_file_input, force_populate=True)
            # If still empty, validation will catch it, but log the failure here too.
            if not self.gemini_file_input.text().strip():
                 logger.error("Failed to auto-populate final filename on start trigger.")
                 # No need to show error here, validation will handle it immediately after.

        # Validate all inputs
        if not self.validate_inputs():
            return

        # Get validated input values
        url = self.url_input.text().strip()
        gemini_output_lang = self.language_input.text().strip() or "English" # Get Gemini Output Lang
        self.selected_model_name = self.model_combo.currentText() # Ensure internal state matches UI
        self.current_chunk_size = self.chunk_size_slider.value()
        self.selected_category = self.category_combo.currentText()
        final_gemini_file_path = self.gemini_file_input.text().strip()
        
        # ---> START: PARSE TRANSCRIPT LANGUAGES <---
        transcript_langs_str = self.transcript_lang_input.text().strip()
        preferred_transcript_languages: List[str] = [] # Initialize an empty list

        # Check if the user actually typed something
        if transcript_langs_str:
            # Split the string by commas
            # For each piece, remove leading/trailing whitespace and convert to lowercase
            # Only keep pieces that are not empty after stripping
            potential_langs = [lang.strip().lower() for lang in transcript_langs_str.split(',') if lang.strip()]

            # Check if the splitting resulted in any valid codes
            if potential_langs:
                preferred_transcript_languages = potential_langs # Use the user's list
            else:
                # Handle cases like input being only "," or " , "
                logger.warning(f"Invalid transcript language input '{transcript_langs_str}', falling back to default: {DEFAULT_TRANSCRIPT_LANGUAGES}")
                preferred_transcript_languages = DEFAULT_TRANSCRIPT_LANGUAGES # Use default
                self.transcript_lang_input.setText(", ".join(preferred_transcript_languages)) # Show default in UI
        else:
            # Input box was empty, use default
            preferred_transcript_languages = DEFAULT_TRANSCRIPT_LANGUAGES # Use default
            self.transcript_lang_input.setText(", ".join(preferred_transcript_languages)) # Show default in UI

        # Store the final list in the main window's state (optional but can be useful)
        self.preferred_transcript_languages = preferred_transcript_languages
        # ---> END: PARSE TRANSCRIPT LANGUAGES <---
        # Generate intermediate file path (use .txt extension)
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            logger.warning("__file__ not defined, using current working directory '.' for intermediate file.")
            script_dir = "."
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_category = re.sub(r'\W+', '_', self.selected_category.lower()) if self.selected_category else "default"
        intermediate_filename = f"{safe_category}_{timestamp}_intermediate.txt" # Use .txt
        intermediate_file_path = os.path.join(script_dir, intermediate_filename)
        self.current_intermediate_file_path = intermediate_file_path # Store for potential deletion

        # Log parameters for the run
        logger.info("="*20 + " Starting New Processing Run " + "="*20)
        logger.info(f"  URL: {url}")
        logger.info(f"  Transcript Languages: {preferred_transcript_languages}") # Log the list being used
        logger.info(f"  Gemini Output Language: {gemini_output_lang}")
        logger.info(f"  Refinement Style: {self.selected_category}")
        logger.info(f"  Gemini Model: {self.selected_model_name}")
        logger.info(f"  Chunk Size: {self.current_chunk_size}")
        logger.info(f"  Intermediate File: {intermediate_file_path}")
        logger.info(f"  Final Output File: {final_gemini_file_path}")
        logger.info("="*55)


        # Update UI for processing state
        self.set_processing_state(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")
        self.status_display.clear()
        self.update_status("Starting process...")

        transcript_langs = DEFAULT_TRANSCRIPT_LANGUAGES # Currently hardcoded, could be UI input

        # Create and start the extraction thread
        # Clean up any potentially lingering old thread reference first
        if self.extraction_thread:
            logger.warning("An old extraction thread reference existed on start. Discarding.")
            self.extraction_thread = None # Just clear reference, let garbage collector handle if needed

        self.extraction_thread = TranscriptExtractionThread(
            url=url,
            output_file=intermediate_file_path,
            preferred_languages=preferred_transcript_languages, # Pass the list we just created
            parent=self
        )
        # Connect signals
        self.extraction_thread.progress_update.connect(self._update_extraction_progress)
        self.extraction_thread.status_update.connect(self.update_status)
        self.extraction_thread.extraction_complete.connect(self._on_extraction_complete)
        self.extraction_thread.error_occurred.connect(self._handle_error)
        self.extraction_thread.finished_signal.connect(self._on_extraction_thread_finished)
        self.extraction_thread.finished.connect(self.extraction_thread.deleteLater) # Qt cleanup

        self.update_status(f"Starting transcript extraction for URL: {url[:60]}...")
        self.progress_bar.setFormat("Extracting: 0%")
        self.extraction_thread.start()


    def cancel_processing(self):
        """Requests cancellation of the ongoing processing task."""
        if not self.is_processing:
            logger.debug("Cancel requested but not currently processing.")
            return

        self.update_status("<font color='#e67e22'>Cancellation requested by user...</font>")
        logger.info("User requested cancellation.")
        self.cancel_button.setEnabled(False) # Prevent multiple clicks
        self.progress_bar.setFormat("Cancelling...")
        QApplication.processEvents() # Update UI immediately

        # Signal threads to stop
        if self.extraction_thread and self.extraction_thread.isRunning():
            logger.debug("Requesting stop for extraction thread.")
            self.extraction_thread.stop()
        if self.gemini_thread and self.gemini_thread.isRunning():
            logger.debug("Requesting stop for Gemini thread.")
            self.gemini_thread.stop()

        # UI state reset is handled by the threads' finished signals
        # Clear intermediate path variable immediately on cancel request
        self.current_intermediate_file_path = None

    def select_default_output_directory(self):
        """Opens a dialog to select and save the default output directory."""
        current_dir = self.default_output_directory or DEFAULT_OUTPUT_DIR_FALLBACK
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Default Directory for Final Outputs",
            current_dir, # Start browser in the current default directory
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if selected_dir: # Proceed only if a directory was selected (not cancelled)
            # Validate the selected directory
            if os.path.isdir(selected_dir) and os.access(selected_dir, os.W_OK | os.R_OK):
                self.default_output_directory = selected_dir
                self.default_output_dir_input.setText(selected_dir)
                self.default_output_dir_input.setCursorPosition(0) # Show start of path
                logger.info(f"Default output directory set to: {selected_dir}")
                self.update_status(f"Default output directory updated: {selected_dir}")
                # Persist the change to config file
                self._save_current_config()
            else:
                logger.warning(f"Selected default directory is invalid or lacks read/write permissions: {selected_dir}")
                self._show_validation_error(f"The selected directory is invalid or lacks read/write permissions:\n{selected_dir}")

    def _save_current_config(self):
        """Saves the current prompts and default directory to config.json."""
        if not save_config(self.prompts_config, self.default_output_directory, DEFAULT_CONFIG_FILE):
             self._show_validation_error(f"Failed to save configuration to {DEFAULT_CONFIG_FILE}.\nPlease check file permissions or logs for details.")
             self.update_status(f"<font color='{COLOR_ACCENT2}'>ERROR: Failed to save configuration changes!</font>")
        else:
             logger.info("Configuration saved successfully after update.")


    def select_gemini_output_file(self):
        """Opens file dialog to select/confirm the final Gemini output file path."""
        self._select_output_file("Select/Confirm Final Gemini Output File (.md)", self.gemini_file_input)

    def _select_output_file(self, title: str, field: QLineEdit, force_populate: bool = False):
        """
        Handles the QFileDialog logic for selecting the final output file.
        Suggests an automated filename based on URL/Style/Timestamp.
        If force_populate is True, it sets the field directly without showing the dialog.
        """
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog # Uncomment for non-native dialog if needed

        # --- Generate Suggested Filename ---
        suggested_name = "gemini_output.md" # Basic fallback
        try:
            current_url = self.url_input.text().strip()
            # Sanitize category name for use in filename
            safe_category = re.sub(r'[\s\\/*?"<>|]+', '_', self.selected_category) if self.selected_category else "DefaultStyle"
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            url_title_part = "Output" # Default prefix

            if current_url:
                is_playlist = "list=" in current_url and ("youtube.com/" in current_url or "youtu.be/" in current_url)
                if is_playlist:
                    url_title_part = "Playlist" # Default if title fetch fails or is slow
                    try:
                        # Attempt to get playlist title (can block, use with caution)
                        pl = Playlist(current_url)
                        if pl and pl.title:
                            # Sanitize playlist title: remove invalid chars, limit length
                            safe_title = re.sub(r'[<>:"/\\|?*\s\.]+', '_', pl.title) # More aggressive sanitization
                            safe_title = safe_title[:40].strip('_') # Limit length and remove leading/trailing underscores
                            if safe_title: url_title_part = safe_title
                            logger.debug(f"Using sanitized playlist title for suggestion: {url_title_part}")
                        else:
                            logger.debug("Playlist object created but title was empty.")
                    except Exception as e:
                        # Log error but proceed with default "Playlist" prefix
                        logger.warning(f"Could not fetch playlist title for filename suggestion: {e}")
                else: # Assume single video
                     url_title_part = "Video"
                     # Getting single video title here is too slow for a responsive file dialog

            suggested_name = f"{url_title_part}_{safe_category}_{timestamp}.md"

        except Exception as e:
            logger.error(f"Error occurred while generating suggested filename: {e}", exc_info=True)
            # Fallback to a simpler timestamped name if generation fails
            suggested_name = f"ErrorGenName_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        # --- End Suggested Filename Generation ---

        # Determine starting directory for the dialog
        start_dir = self.default_output_directory or DEFAULT_OUTPUT_DIR_FALLBACK
        if not os.path.isdir(start_dir):
            logger.warning(f"Default output directory '{start_dir}' is invalid or doesn't exist. Falling back to home directory.")
            start_dir = DEFAULT_OUTPUT_DIR_FALLBACK

        # --- Show Dialog or Populate Field ---
        full_suggested_path = os.path.join(start_dir, suggested_name)

        if force_populate:
            # Called from start_processing if field was empty; directly set the field text
            field.setText(full_suggested_path)
            logger.info(f"Auto-populated final output field with suggested path: {full_suggested_path}")
            return # Skip showing the dialog

        # Default behavior: show the save file dialog
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            title,
            full_suggested_path, # Pre-fill dialog with suggested path
            "Markdown Files (*.md);;All Files (*)", # Filter for Markdown files
            options=options
        )

        if file_path:
            # User selected a path, update the QLineEdit field
            if not file_path.lower().endswith(".md"):
                file_path += ".md" # Ensure .md extension
            field.setText(file_path)
            logger.debug(f"Final output file selected/confirmed by user: {file_path}")
        else:
            # User cancelled the dialog
            logger.debug("File selection dialog was cancelled by the user.")
            # The field remains unchanged (might be empty or hold a previous value)


    def center(self):
        """Centers the main window on the primary screen."""
        try:
            frame_geom = self.frameGeometry()
            screen = QApplication.primaryScreen()
            if screen:
                center_point = screen.availableGeometry().center()
                frame_geom.moveCenter(center_point)
                self.move(frame_geom.topLeft())
            else:
                logger.warning("Could not get primary screen information to center window.")
        except Exception as e:
            logger.warning(f"Error centering window: {e}")

    def closeEvent(self, event):
        """Handles the window close event, confirms exit if processing."""
        logger.info("Close event triggered.")
        if self.is_processing:
             reply = QMessageBox.question(self, 'Confirm Exit',
                                          "Processing is currently in progress.\nAre you sure you want to exit?",
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
             if reply == QMessageBox.Yes:
                 logger.info("User confirmed exit during processing. Attempting cancellation.")
                 self.cancel_processing() # Request cancellation
                 event.accept() # Allow closing
             else:
                 logger.info("User cancelled exit request.")
                 event.ignore() # Prevent closing
        else:
            logger.info("Exiting application cleanly.")
            event.accept() # Allow closing


# --- Application Entry Point ---
if __name__ == "__main__":
    main_logger = logging.getLogger(__name__) # Use root logger configured earlier
    main_logger.info(f"{'='*20} Application Starting {'='*20}")
    main_logger.info(f"Log file located at: {os.path.abspath(log_file)}")
    # Ensure required dependencies are installed via requirements.txt or manually

    # Enable High DPI scaling if available
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        main_logger.debug("HighDpiScaling attribute set.")
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        main_logger.debug("UseHighDpiPixmaps attribute set.")

    app = QApplication(sys.argv)

    try:
        window = MainWindow()
        window.center() # Center window on screen
        window.show()
        main_logger.info("Main window created and shown.")
        exit_code = app.exec_()
        main_logger.info(f"Application event loop finished. Exit code: {exit_code}")
        main_logger.info(f"{'='*20} Application Exiting {'='*21}")
        sys.exit(exit_code)
    except Exception as e:
        # Catch critical errors during application setup or execution
        main_logger.critical(f"Unhandled critical exception in main execution block: {e}", exc_info=True)
        # Attempt to show a critical error message box
        try:
            QMessageBox.critical(None, "Fatal Error",
                                 f"A critical error occurred and the application must close:\n\n{e}\n\n"
                                 f"Please check the log file for details:\n'{os.path.abspath(log_file)}'.",
                                 QMessageBox.Ok)
        except Exception as mb_e:
            # Log if even the message box fails
            main_logger.error(f"Could not display the final critical error message box: {mb_e}")
        sys.exit(1) # Exit with error code
