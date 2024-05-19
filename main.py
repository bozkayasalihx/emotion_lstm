#!/usr/bin/env python3

import os
import pandas as pd
import librosa
import numpy as np
import torch
import torch.nn as nn

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
        data,
        columns=["emotion", "emotion intensity", "gender", "path"],
    )


SAMPLE_RATE = 48000


def extract_mfcc(file_path: str):
    audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5, sr=SAMPLE_RATE)
    signal = np.zeros(
        (
            int(
                SAMPLE_RATE * 3,
            )
        )
    )
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


RANDOM_STATE = 42

# returns [x_train, y_train, x_test, y_test, x_val, y_val];


def split_data(data: pd.DataFrame):
    X_train = data.sample(n=1147, random_state=RANDOM_STATE)
    X_train["data"] = X_train["path"].apply(lambda x: extract_mfcc(x))
    Y_train = X_train[["emotion"]]
    remaining_subset = data.drop(index=X_train.index)

    X_val = remaining_subset.sample(n=143, random_state=RANDOM_STATE)
    X_val["data"] = X_val["path"].apply(lambda x: extract_mfcc(x))
    Y_val = X_val[["emotion"]]
    remaining_subset = remaining_subset.drop(index=X_val.index)

    remaining_subset["data"] = remaining_subset["path"].apply(lambda x: extract_mfcc(x))
    Y_test = remaining_subset[["emotion"]]

    return X_train, Y_train, X_val, Y_val, remaining_subset, Y_test


def tokenize(data: pd.DataFrame):
    EMOTIONS = {
        "suprise": 0,
        "neural": 1,
        "calm": 2,
        "happy": 3,
        "sad": 4,
        "angry": 5,
        "fear": 6,
        "disgust": 7,
    }
    data.loc[:, ("emotion")] = data.loc[:, ("emotion")].apply(
        lambda x: 0 if EMOTIONS[x] == 8 else EMOTIONS[x]
    )
    return np.array(data.to_numpy().reshape((data.shape[0],)), dtype=np.int8)


def make_validate_fnc(model, loss_fnc):
    def validate(X, Y):
        with torch.no_grad():
            model.eval()
            output_logits, output_softmax, attention_weights_norm = model(X)
            predictions = torch.argmax(output_softmax, dim=1)
            accuracy = torch.sum(Y == predictions) / float(len(Y))
            loss = loss_fnc(output_logits, Y)
        return loss.item(), accuracy * 100, predictions

    return validate


def make_train_step(model, loss_fnc, optimizer):
    def train_step(X, Y):
        model.train()
        output_logits, outpout_softmax, atttention_weights_norm = model(X)
        predictions = torch.argmax(outpout_softmax, dim=1)
        accuracy = torch.sum(Y == predictions) / float(len(Y))
        loss = loss_fnc(output_logits, Y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        return loss.item(), accuracy * 100

    return train_step


class EmotionLSTM(nn.Module):
    def __init__(self, num_of_emotions):
        super().__init__()
        # conv block
        self.conv2Dblock = nn.Sequential(
            # 1. conv block
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
            # 2. conv block
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
            # 3. conv block
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
            # 4. conv block
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        # LSTM block
        self.lstm_maxpool = nn.MaxPool2d(kernel_size=[2, 4], stride=[2, 4])
        hidden_size = 128
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout_lstm = nn.Dropout(0.1)
        self.attention_linear = nn.Linear(
            2 * hidden_size,
            1,
        )
        # Linear softmax layer
        self.out_linear = nn.Linear(4 * hidden_size, num_of_emotions)
        self.dropout_linear = nn.Dropout(p=0)
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        conv_embedding = self.conv2Dblock(x)
        conv_embedding = torch.flatten(conv_embedding, start_dim=1)
        x_reduced = self.lstm_maxpool(x)
        x_reduced = torch.squeeze(x_reduced, 1)
        x_reduced = x_reduced.permute(0, 2, 1)
        lstm_embedding, (h, c) = self.lstm(x_reduced)
        attention_weights = torch.stack([self.attention_linear(lstm_embedding[:, t, :]) for t in range(lstm_embedding.size(1))], dim=1)
        attention_weights_norm = nn.functional.softmax(attention_weights, dim=1)
        attention = torch.bmm(attention_weights_norm.permute(0, 2, 1), lstm_embedding)
        attention = torch.squeeze(attention, 1)
        complete_embedding = torch.cat([conv_embedding, attention], dim=1)
        output_logits = self.out_linear(complete_embedding)
        output_softmax = self.out_softmax(output_logits)
        return output_logits, output_softmax, attention_weights_norm



def transform_data(data: pd.DataFrame):
    from sklearn.preprocessing import StandardScaler

    mel_data = []
    for _, item in data.iterrows():
        mel_spec = getMELspec(item["data"], sample_rate=SAMPLE_RATE)
        mel_data.append(mel_spec)

    scaler = StandardScaler()
    mel_data = np.expand_dims(np.stack(mel_data, axis=0), 1)
    b, c, h, w = mel_data.shape
    mel_data = np.reshape(mel_data, newshape=(b, -1))
    mel_data = scaler.fit_transform(mel_data)
    mel_data = np.reshape(mel_data, newshape=(b, c, h, w))

    return mel_data


def train(
    xtrain_data: pd.DataFrame,
    ytrain_data: np.ndarray,
    X_val: pd.DataFrame,
    Y_val: np.ndarray,
):
    EPOCHS = 15000
    BATCH_SIZE = 32
    DATASET_SIZE = xtrain_data.shape[0]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = EmotionLSTM(num_of_emotions=len(EMOTIONS))
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, 
        # weight_decay=1e-3, momentum=0.8
    )

    train_step = make_train_step(model, loss_fnc=loss_function, optimizer=optimizer)
    validate = make_validate_fnc(model, loss_function)
    train_xdata = transform_data(xtrain_data)
    X_val = transform_data(X_val)
    losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        idx = np.random.permutation(DATASET_SIZE)
        train_xdata = train_xdata[idx, :, :, :]
        ytrain_data = ytrain_data[idx]

        epoch_acc = 0
        epoch_loss = 0

        iters = int(DATASET_SIZE / BATCH_SIZE)
        for i in range(iters):
            batch_start = i * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, DATASET_SIZE)
            actual_batch_size = batch_end - batch_start
            X = train_xdata[batch_start:batch_end, :, :, :]
            Y = ytrain_data[batch_start:batch_end]
            X_tensor = torch.tensor(X, dtype=torch.float, device=device)
            Y_tensor = torch.tensor(Y, dtype=torch.long, device=device)
            loss, acc = train_step(X_tensor, Y_tensor)
            epoch_acc += acc * actual_batch_size / train_xdata.shape[0]
            epoch_loss += loss * actual_batch_size / train_xdata.shape[0]
            print(f"\r Epoch {epoch}: iteration {i}/{iters}", end="")

        X_val_tensor = torch.tensor(X_val, device=device, dtype=torch.float)
        Y_val_tensor = torch.tensor(Y_val, device=device, dtype=torch.long)
        val_loss, val_acc, _ = validate(X_val_tensor, Y_val_tensor)
        losses.append(epoch_loss)
        val_losses.append(val_loss)
        print("")
        print(
            f"Epoch {epoch} --> loss:{epoch_loss:.4f}, acc:{epoch_acc:.2f}%, val_loss:{val_loss:.4f}, val_acc:{val_acc:.2f}%"
        )
    print("saving model")
    save_model(model)


def save_model(model):
    MODEL_PATH = os.path.join(os.getcwd(), "models")
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, "emotion_lstm.pt"))
    print("model saved")


if __name__ == "__main__":
    DATA_PATH = os.getenv("DATA_PATH")
    if DATA_PATH is None:
        print("DATA_PATH should be given")
    else:
        df = prep_data(DATA_PATH)
        X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(df)
        train(X_train, tokenize(Y_train), X_val, tokenize(Y_val));
