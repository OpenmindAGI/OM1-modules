import argparse
import base64
import json
import logging
import shutil
import subprocess
import threading
import time
from queue import Empty, Queue
from typing import Callable, Dict, Optional

import pyaudio
import requests

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class AudioOutputStream:
    """
    A class for managing audio output and text-to-speech (TTS) conversion.

    Parameters
    ----------
    url : str
        The URL endpoint for the text-to-speech service
    rate : int, optional
        The sampling rate in Hz for audio output (default: 8000)
    device : int, optional
        The output device index. If None, uses the first available output device
        (default: None)
    device_name: str, optional
        The output device name. If None, uses the first available output device
        (default: None)
    tts_state_callback : Optional[Callable], optional
        A callback function to receive TTS state changes (active/inactive)
        (default: None)
    headers : Optional[Dict[str, str]], optional
        Additional headers to include in the HTTP request (default: None)
    """

    def __init__(
        self,
        url: str,
        rate: int = 8000,
        device: int = None,
        device_name: str = None,
        tts_state_callback: Optional[Callable] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self._url = url
        self._rate = rate
        self._device = device
        self._device_name = device_name

        # Process headers
        self._headers = headers or {}
        if "Content-Type" not in self._headers:
            self._headers["Content-Type"] = "application/json"

        # Callback for TTS state
        self._tts_state_callback = tts_state_callback

        # Pending requests queue
        self._pending_requests: Queue[Optional[str]] = Queue()

        # Initialize audio interface
        self._pyaudio_interface = pyaudio.PyAudio()

        self.running: bool = True
        self._last_audio_time = time.time()

        if self._device is not None and self._device_name is not None:
            logger.error("Only one of device or device_name can be specified")
            raise ValueError("Only one of device or device_name can be specified")

        # Find a suitable output device
        self._device = self._select_output_device()

    def _select_output_device(self) -> int:
        """
        Select and validate audio output device.
        """
        device_count = self._pyaudio_interface.get_device_count()
        logger.info(f"Found {device_count} audio devices")

        if self._device is not None:
            device_info = self._pyaudio_interface.get_device_info_by_index(self._device)
            if device_info["maxOutputChannels"] == 0:
                raise ValueError("Selected output device has no output channels")
            logger.info(
                f"Selected output device: {device_info['name']} ({self._device})"
            )
            return self._device

        if self._device_name is not None:
            available_devices = []
            for i in range(device_count):
                device_info = self._pyaudio_interface.get_device_info_by_index(i)
                if device_info["maxOutputChannels"] > 0:
                    device_name = device_info["name"]
                    available_devices.append({"name": device_name, "index": i})
                    if self._device_name.lower() in device_name.lower():
                        logger.info(f"Found device by name: {device_name} ({i})")
                        return i
            raise ValueError(
                f"No output device found with name {self._device_name}. Available devices: {available_devices}"
            )

        default_device_index = self._pyaudio_interface.get_default_output_device_info()[
            "index"
        ]
        device_info = self._pyaudio_interface.get_device_info_by_index(
            default_device_index
        )
        if device_info["maxOutputChannels"] == 0:
            raise ValueError(
                f"Default output device {device_info['name']} has no output channels"
            )

        logger.info(
            f"Using default output device: {device_info['name']} ({default_device_index})"
        )
        return default_device_index

    def set_tts_state_callback(self, callback: Callable):
        """
        Set a callback function for TTS state changes.

        Parameters
        ----------
        callback : Callable
            Function to be called when TTS state changes (active/inactive)
        """
        self._tts_state_callback = callback

    def add_request(self, audio_request: Dict[str, str]):
        """
        Add request to the TTS processing queue.

        Parameters
        ----------
        audio_request : Dict[str, str]
            Request to be processed by the TTS service
        """
        self._pending_requests.put(audio_request)

    def _process_audio(self):
        """
        Process the TTS queue and play audio output.

        Makes HTTP requests to the TTS service, converts responses to audio,
        and plays them through the audio device.
        """
        while self.running:
            try:
                tts_request = self._pending_requests.get_nowait()
                response = requests.post(
                    self._url,
                    data=json.dumps(tts_request),
                    headers=self._headers,
                    timeout=(5, 15),
                )
                logger.info(f"Received TTS response: {response.status_code}")
                if response.status_code == 200:
                    result = response.json()
                    if "response" in result:
                        audio_data = result["response"]
                        self._write_audio(audio_data)
                        logger.debug(f"Processed TTS request: {tts_request}")
                else:
                    logger.error(
                        f"TTS request failed with status code {response.status_code}: {response.text}. Request details: {tts_request}"
                    )
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                continue

    def _create_silence_audio(self, duration_ms: int = 500) -> bytes:
        """
        Create silent audio data.

        Parameters
        ----------
        duration_ms : int
            Duration of silence in milliseconds

        Returns
        -------
        bytes
            Base64 encoded silent audio data
        """
        samples = int(self._rate * duration_ms / 1000)
        silence_bytes = b"\x00" * (samples * 2)
        return base64.b64encode(silence_bytes)

    def _play_keepalive_sound(self):
        """Play a very brief silent audio to keep Bluetooth speakers awake."""
        silence_audio = self._create_silence_audio(100)
        self._write_audio_raw(silence_audio, is_keepalive=True)

    def _keepalive_worker(self):
        """Background thread to play keepalive sounds every 60 seconds."""
        while self.running:
            current_time = time.time()
            if current_time - self._last_audio_time >= 60:
                self._play_keepalive_sound()
                self._last_audio_time = current_time
            time.sleep(10)

    def _write_audio(self, audio_data: bytes):
        """
        Write audio data to the output stream with Bluetooth optimization.

        Parameters
        ----------
        audio_data : bytes
            The audio data to be written to the output stream
        """
        self._last_audio_time = time.time()

        silence_prefix = self._create_silence_audio(500)
        audio_bytes = base64.b64decode(silence_prefix) + base64.b64decode(audio_data)

        self._write_audio_raw(base64.b64encode(audio_bytes))

    def _write_audio_raw(self, audio_data: bytes, is_keepalive: bool = False):
        """
        Write raw audio data to the output stream.

        Parameters
        ----------
        audio_data : bytes
            The audio data to be written to the output stream
        is_keepalive : bool
            Whether this is a keepalive sound (suppresses callbacks)
        """
        audio_bytes = base64.b64decode(audio_data)

        if not is_installed("ffplay"):
            message = (
                "ffplay from ffmpeg not found, necessary to play audio. "
                "On mac you can install it with 'brew install ffmpeg'. "
                "On linux and windows you can install it from https://ffmpeg.org/"
            )
            logger.error(message)
            return

        if not is_keepalive:
            self._tts_callback(True)

        args = [
            "ffplay",
            "-autoexit",
            "-",
            "-nodisp",
            # It is not good at supporting audio_device_index from pyaudio
            # Reading the list from ffplay doesn't work either
            # "-audio_device_index",
            # str(self._device),
        ]
        proc = subprocess.Popen(
            args=args,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        proc.communicate(input=audio_bytes)
        exit_code = proc.poll()

        if exit_code != 0 and not is_keepalive:
            logger.error(f"Error playing audio: {exit_code}")

        if not is_keepalive:
            self._tts_callback(False)

    def _tts_callback(self, is_active: bool):
        """
        Invoke the TTS state callback if set.

        Parameters
        ----------
        is_active : bool
            Whether TTS is currently active
        """
        if self._tts_state_callback:
            self._tts_state_callback(is_active)

    def start(self):
        """
        Start the audio processing thread.

        Initializes a daemon thread for processing the TTS queue.
        """
        process_thread = threading.Thread(target=self._process_audio)
        process_thread.daemon = True
        process_thread.start()

        keepalive_thread = threading.Thread(target=self._keepalive_worker)
        keepalive_thread.daemon = True
        keepalive_thread.start()

    def run_interactive(self):
        """
        Run an interactive console for text-to-speech conversion.

        Allows users to input text for TTS conversion until 'quit' is entered
        or KeyboardInterrupt is received.
        """
        logger.info(
            "Running interactive audio output stream. Please enter text for TTS conversion."
        )
        try:
            while self.running:
                user_input = input()
                if user_input.lower() == "quit":
                    break
                self.add_request({"text": user_input})
        except KeyboardInterrupt:
            self.stop()
        finally:
            self.stop()

    def stop(self):
        """
        Stop the audio output stream and cleanup resources.
        """
        self.running = False


def is_installed(lib_name: str) -> bool:
    return shutil.which(lib_name) is not None


def main():
    """
    Main function for running the audio output stream.
    """
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tts-url", type=str, required=True, help="URL for the TTS service"
    )
    parser.add_argument("--device", type=int, default=None, help="Output device index")
    parser.add_argument(
        "--rate", type=int, default=8000, help="Audio output rate in Hz"
    )
    args = parser.parse_args()

    audio_output = AudioOutputStream(args.tts_url, device=args.device, rate=args.rate)
    audio_output.start()
    audio_output.run_interactive()
    audio_output.stop()


if __name__ == "__main__":
    main()
