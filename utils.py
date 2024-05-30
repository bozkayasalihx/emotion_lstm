import torch
import librosa
import numpy as np
import os


def get_device() -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device


def extract_mfcc(file_path: str, sr: int) -> np.ndarray:
    audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5, sr=sr)
    signal = np.zeros((int( sample_rate* 3)))
    signal[: len(audio)] = audio
    return signal

def extract_mel_spectogram(data: np.ndarray, sr: int) -> np.ndarray:
    mel_spec = librosa.feature.melspectrogram(
        y=data,
        sr=sr,
        n_fft=1024,
        win_length=512,
        window="hamming",
        hop_length=256,
        n_mels=128,
        fmax=sr/ 2,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def extract_features(path: str, sr: int):
    result = np.array([])
    audio_data, sample_rate = librosa.load(path, duration=2.5, offset=0.6, sr=sr)
    mfcc = extract_mfcc(audio_data, sample_rate)
    result = np.hstack((result, mfcc))  
    mel_spec = extract_mel_spectogram(audio_data, sample_rate)
    result = np.hstack((result, mel_spec))
    return result

    

def save_model(model):
    MODEL_PATH = os.path.join(os.getcwd(), "models")
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, "emotion_lstm.pt"))
    print("model saved")
