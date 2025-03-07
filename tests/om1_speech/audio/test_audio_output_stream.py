import base64
import json
import time
from queue import Queue
from unittest.mock import Mock, patch

import pytest

from om1_speech import AudioOutputStream


@pytest.fixture
def mock_pyaudio():
    with patch("pyaudio.PyAudio") as mock:
        # Setup default device info
        device_info = {
            "name": "Test Device",
            "maxOutputChannels": 2,
            "defaultSampleRate": 44100,
            "index": 0,
        }

        # Mock device count and default device info
        mock.return_value.get_device_count.return_value = 2
        mock.return_value.get_device_info_by_index.return_value = device_info
        mock.return_value.get_default_output_device_info.return_value = device_info

        yield mock


@pytest.fixture
def mock_subprocess():
    with patch("subprocess.Popen") as mock:
        # Mock process and communicate
        mock_process = Mock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.poll.return_value = 0
        mock.return_value = mock_process
        yield mock


@pytest.fixture
def mock_requests():
    with patch("requests.post") as mock:
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": base64.b64encode(b"test_audio_data").decode("utf-8")
        }
        mock.return_value = mock_response
        yield mock


@pytest.fixture
def audio_output(mock_pyaudio, mock_requests, mock_subprocess):
    stream = AudioOutputStream(url="http://test-tts-server/tts", rate=16000)
    yield stream
    stream.stop()


def test_initialization(audio_output, mock_pyaudio):
    """Test AudioOutputStream initialization"""
    assert audio_output._rate == 16000
    assert audio_output._url == "http://test-tts-server/tts"
    assert audio_output.running is True
    assert isinstance(audio_output._pending_requests, Queue)

    # Verify PyAudio initialization
    mock_pyaudio.assert_called_once()


def test_device_selection_default(mock_pyaudio):
    """Test default device selection"""
    AudioOutputStream("http://test-tts-server/tts")

    # Verify device enumeration
    mock_pyaudio.return_value.get_device_count.assert_called_once()
    mock_pyaudio.return_value.get_default_output_device_info.assert_called_once()
    mock_pyaudio.return_value.get_device_info_by_index.assert_called()


def test_device_selection_specific(mock_pyaudio):
    """Test specific device selection"""
    AudioOutputStream("http://test-tts-server/tts", device=1)

    # Verify specific device was selected
    mock_pyaudio.return_value.get_device_info_by_index.assert_called_with(1)


def test_device_selection_by_name(mock_pyaudio):
    """Test device selection by name"""
    # Setup multiple devices
    devices = [
        {"name": "Device 1", "maxOutputChannels": 2, "index": 0},
        {"name": "Test Device", "maxOutputChannels": 2, "index": 1},
    ]

    def get_device_by_index(index):
        return devices[index]

    mock_pyaudio.return_value.get_device_info_by_index.side_effect = get_device_by_index

    # Create with device name
    AudioOutputStream("http://test-tts-server/tts", device_name="Test")

    # Verify device enumeration was done to find by name
    assert mock_pyaudio.return_value.get_device_count.call_count >= 1


def test_device_conflicting_params():
    """Test error when both device and device_name are specified"""
    with pytest.raises(
        ValueError, match="Only one of device or device_name can be specified"
    ):
        AudioOutputStream("http://test-tts-server/tts", device=1, device_name="Test")


def test_tts_callback(mock_pyaudio):
    """Test TTS state callback"""
    callback_state = None

    def tts_callback(state):
        nonlocal callback_state
        callback_state = state

    stream = AudioOutputStream(
        "http://test-tts-server/tts", tts_state_callback=tts_callback
    )

    stream._tts_callback(True)
    assert callback_state is True

    stream._tts_callback(False)
    assert callback_state is False

    stream.stop()


def test_audio_processing(audio_output, mock_requests, mock_subprocess):
    """Test audio processing flow"""
    # Create a flag for callback verification
    callback_called = False

    def tts_callback(state):
        nonlocal callback_called
        callback_called = state

    audio_output.set_tts_state_callback(tts_callback)

    # Start processing
    audio_output.start()

    # Add test input
    test_text = "Hello, world!"
    audio_output.add_request({"text": test_text})

    # Wait a bit for processing
    time.sleep(0.1)

    # Verify request was made
    mock_requests.assert_called_with(
        "http://test-tts-server/tts",
        data=json.dumps({"text": test_text}),
        headers={"Content-Type": "application/json"},
        timeout=(5, 15),
    )

    # Verify ffplay was called with correct parameters
    mock_subprocess.assert_called_once()
    args = mock_subprocess.call_args[1]["args"]
    assert "ffplay" in args[0]
    assert "-autoexit" in args


def test_error_handling(audio_output, mock_requests):
    """Test error handling in audio processing"""
    # Make the request fail
    mock_requests.return_value.status_code = 500

    # Start processing
    audio_output.start()

    # Add test input
    audio_output.add_request({"text": "Test error handling"})

    # Wait a bit for processing
    time.sleep(0.1)

    # Verify stream is still running
    assert audio_output.running is True


def test_stop(audio_output):
    """Test stopping the audio output stream"""
    audio_output.start()
    audio_output.stop()

    assert audio_output.running is False


def test_empty_queue_handling(audio_output):
    """Test handling of empty queue"""
    audio_output.start()

    # Wait a bit with empty queue
    time.sleep(0.1)

    # Verify stream is still running
    assert audio_output.running is True


@pytest.mark.parametrize("rate", [8000, 16000, 44100])
def test_different_sample_rates(mock_pyaudio, rate):
    """Test initialization with different sample rates"""
    stream = AudioOutputStream("http://test-tts-server/tts", rate=rate)

    # Verify rate was stored correctly
    assert stream._rate == rate

    stream.stop()


def test_add_multiple_items(audio_output):
    """Test adding multiple items to the queue"""
    items = [{"text": "Test 1"}, {"text": "Test 2"}, {"text": "Test 3"}]

    for item in items:
        audio_output.add_request(item)

    # Verify items were added to queue
    assert audio_output._pending_requests.qsize() == len(items)

    # Verify queue contents
    received_items = []
    while not audio_output._pending_requests.empty():
        received_items.append(audio_output._pending_requests.get())

    assert received_items == items


def test_subprocess_error(audio_output, mock_subprocess, mock_requests):
    """Test handling of subprocess errors"""
    # Make subprocess return error code
    mock_subprocess.return_value.poll.return_value = 1

    # Start processing
    audio_output.start()

    # Add test input
    audio_output.add_request({"text": "Test subprocess error"})

    # Wait a bit for processing
    time.sleep(0.1)

    # Verify stream is still running despite the error
    assert audio_output.running is True
