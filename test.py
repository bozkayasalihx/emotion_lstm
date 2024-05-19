#!/usr/bin/env python3

import os
import pandas as pd
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

EMOTIONS = {
    1: "neural",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fear",
    7: "disgust",
    0: "suprise",
}

EMOTION_INTENSITY = {
    1: "normal",
    2: "strong",
}

RANDOM_STATE = 42
SAMPLE_RATE = 48000


class AudioDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        audio_path = self.dataframe.iloc[idx]["path"]
        label = self.dataframe.iloc[idx]["emotion"]
        audio = extract_mfcc(audio_path)
        mel_spec = getMELspec(audio, SAMPLE_RATE)
        if self.transform:
            mel_spec = self.transform(mel_spec)
        return torch.tensor(mel_spec, dtype=torch.float32), label


def loss_function(predictions, targets):
    return nn.CrossEntropyLoss()(predictions, targets)


def prep_data(file_path: str):
    data = []
    for dir in os.listdir(file_path):
        each_audio_path = os.path.join(file_path, dir)
        for audio_path in os.listdir(each_audio_path):
            full_path = os.path.join(each_audio_path, audio_path)
            path_prefix = audio_path.split(".")[0].split("-")
            emotion = int(path_prefix[2])
            emotion_intense = EMOTION_INTENSITY[int(path_prefix[3])]
            if emotion == 8:
                emotion = 0
            if int(path_prefix[6]) % 2 == 0:
                # its female
                data.append([EMOTIONS[emotion], emotion_intense, "female", full_path])
            else:
                # its male
                data.append([EMOTIONS[emotion], emotion_intense, "male", full_path])

    return pd.DataFrame(
        data, columns=["emotion", "emotion intensity", "gender", "path"]
    )


def extract_mfcc(file_path: str):
    audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5, sr=SAMPLE_RATE)
    signal = np.zeros((int(SAMPLE_RATE * 3),))
    signal[: len(audio)] = audio
    return signal


def getMELspec(audio, sample_rate):
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=1024,
        win_length=512,
        window="hamming",
        hop_length=256,
        n_mels=128,
        fmax=sample_rate / 2,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


class EmotionLSTM(nn.Module):
    def __init__(self, num_of_emotions):
        super(EmotionLSTM, self).__init__()
        self.conv2Dblock = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        self.lstm_maxpool = nn.MaxPool2d(kernel_size=[2, 4], stride=[2, 4])
        hidden_size = 128
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=hidden_size, bidirectional=True, batch_first=True
        )
        self.attention_linear = nn.Linear(2 * hidden_size, 1)
        self.out_linear = nn.Linear(4 * hidden_size, num_of_emotions)
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        conv_embedding = self.conv2Dblock(x)
        conv_embedding = torch.flatten(conv_embedding, start_dim=1)
        x_reduced = self.lstm_maxpool(x)
        x_reduced = torch.squeeze(x_reduced, 1)
        x_reduced = x_reduced.permute(0, 2, 1)
        lstm_embedding, (h, c) = self.lstm(x_reduced)
        attention_weights = torch.stack(
            [
                self.attention_linear(lstm_embedding[:, t, :])
                for t in range(lstm_embedding.size(1))
            ],
            dim=1,
        )
        attention_weights_norm = nn.functional.softmax(attention_weights, dim=1)
        attention = torch.bmm(attention_weights_norm.permute(0, 2, 1), lstm_embedding)
        attention = torch.squeeze(attention, 1)
        complete_embedding = torch.cat([conv_embedding, attention], dim=1)

        output_logits = self.out_linear(complete_embedding)
        output_softmax = self.out_softmax(output_logits)
        return output_logits, output_softmax, attention_weights_norm


def train(model, dataloaders, criterion, optimizer, num_epochs=25, device="cuda:0"):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs, _, _ = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print("Training complete")
    print(f"Best val Acc: {best_acc:4f}")

    model.load_state_dict(best_model_wts)
    return model


def main(data_path):
    df = prep_data(data_path)
    X_train, X_temp, y_train, y_temp = train_test_split(
        df, df["emotion"], test_size=0.2, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    train_dataset = AudioDataset(X_train, transform=scaler.fit_transform)
    val_dataset = AudioDataset(X_val, transform=scaler.transform)
    test_dataset = AudioDataset(X_test, transform=scaler.transform)

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
        "val": DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4),
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EmotionLSTM(num_of_emotions=len(EMOTIONS))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = train(
        model, dataloaders, criterion, optimizer, num_epochs=25, device=device
    )

    save_model(model, "emotion_lstm.pt")


def save_model(model, filename):
    MODEL_PATH = os.path.join(os.getcwd(), "models")
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, filename))
    print("model saved")


if __name__ == "__main__":
    DATA_PATH = os.getenv("DATA_PATH")
    if DATA_PATH is None:
        print("DATA_PATH should be given")
    else:
        main(DATA_PATH)
