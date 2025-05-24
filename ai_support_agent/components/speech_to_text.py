import asyncio
from datetime import datetime
from typing import AsyncGenerator, Optional, Dict, Any, Callable
from utils.models import TranscriptEntry, Speaker
import os
import logging
import threading
import pyaudio
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import assemblyai as aai
except ImportError:
    logger.error(
        "AssemblyAI package not found. Please install it with: pip install assemblyai"
    )
    raise


class SpeechToTextService:
    """Real-time Speech-to-Text service using AssemblyAI - Single speaker mode"""

    def __init__(self):
        self.is_running = False
        self.transcriber: Optional[aai.RealtimeTranscriber] = None
        self.microphone_stream: Optional[aai.extras.MicrophoneStream] = None
        self._audio = pyaudio.PyAudio()
        self._streaming_task = None
        self._entry_callback: Optional[Callable] = None

        aai.settings.api_key = Config.ASSEMBLYAI_API_KEY

    def _get_default_input_device(self) -> int:
        """Get the default input device index"""
        try:
            default_device = self._audio.get_default_input_device_info()
            device_index = default_device.get("index")
            if not isinstance(device_index, int):
                raise ValueError("Invalid device index")
            logger.info(
                f"Using default input device: {default_device.get('name')}"
            )
            return device_index
        except Exception as e:
            logger.warning(f"Could not get default input device: {e}")
            # Try to find any working input device
            for i in range(self._audio.get_device_count()):
                device_info = self._audio.get_device_info_by_index(i)
                max_input_channels = device_info.get("maxInputChannels", 0)
                if (
                    isinstance(max_input_channels, (int, float))
                    and max_input_channels > 0
                ):
                    logger.info(
                        f"Using input device: {device_info.get('name')}"
                    )
                    return i
            raise RuntimeError("No input devices found")

    def set_entry_callback(self, callback: Callable[[TranscriptEntry], None]):
        """Set callback for when new entries are available"""
        self._entry_callback = callback

    async def start_transcription_with_callback(self):
        """Start transcription service with callback approach"""
        logger.info("Starting AssemblyAI transcription service with callback")
        self.is_running = True

        try:
            # Create the Real-Time transcriber
            self.transcriber = aai.RealtimeTranscriber(
                sample_rate=44_100,
                on_data=self._handle_transcript,
                on_error=self._handle_error,
                on_open=self._handle_open,
                on_close=self._handle_close,
            )

            if not self.transcriber:
                raise RuntimeError("Failed to create transcriber")

            # Start the connection
            self.transcriber.connect()

            # Open a microphone stream with specific device
            try:
                device_index = self._get_default_input_device()
                self.microphone_stream = aai.extras.MicrophoneStream(
                    device_index=device_index,
                    sample_rate=44_100,
                )
            except Exception as e:
                logger.error(f"Failed to initialize microphone: {e}")
                raise RuntimeError(f"Could not initialize microphone: {e}")

            if not self.microphone_stream:
                raise RuntimeError("Failed to create microphone stream")

            # Start streaming in a separate task
            self._streaming_task = asyncio.create_task(self._stream_audio())

            # Keep running while transcription is active
            while self.is_running:
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            self.stop_transcription()
        finally:
            if self._streaming_task and not self._streaming_task.done():
                self._streaming_task.cancel()

    async def _stream_audio(self):
        """Stream audio in a separate task"""
        if not self.transcriber or not self.microphone_stream:
            logger.error("Transcriber or microphone stream not initialized")
            self.stop_transcription()
            return

        try:
            self.transcriber.stream(self.microphone_stream)
        except Exception as e:
            logger.error(f"Audio streaming error: {e}")
            self.stop_transcription()

    def stop_transcription(self):
        """Stop the transcription service"""
        self.is_running = False
        if self.transcriber:
            try:
                self.transcriber.close()
            except Exception as e:
                logger.error(f"Error closing transcriber: {e}")
        if self._audio:
            try:
                self._audio.terminate()
            except Exception as e:
                logger.error(f"Error terminating audio: {e}")
        logger.info("Transcription service stopped")

    def _handle_transcript(self, transcript: aai.RealtimeTranscript):
        """Handle incoming transcript data - treat all speech as from speaker"""
        if not transcript.text:
            return

        if isinstance(transcript, aai.RealtimeFinalTranscript):
            # Treat all speech as from a single speaker (could be customer or agent)
            # The LLM will figure out who is speaking based on context

            # Create the transcript entry
            entry = TranscriptEntry(
                speaker=Speaker.SPEAKER,  # Use the enum directly
                text=transcript.text,
                timestamp=datetime.now(),
            )
            logger.info(f"Generated entry: {entry.speaker.value}: {entry.text}")

            # Call the callback if set
            if self._entry_callback:
                try:
                    self._entry_callback(entry)
                except Exception as e:
                    logger.error(f"Error in entry callback: {e}")

    def _handle_error(self, error: aai.RealtimeError):
        """Handle transcription errors"""
        logger.error(f"Transcription error: {error}")

    def _handle_open(self, session_opened: aai.RealtimeSessionOpened):
        """Handle session open"""
        logger.info(f"Session opened: {session_opened.session_id}")

    def _handle_close(self):
        """Handle session close"""
        logger.info("Session closed")
