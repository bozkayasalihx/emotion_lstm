#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import torch
import random
from model import EmotionLSTM, loss_function
from tokenizer import tokenizer
from utils import get_device, extract_mfcc, get_mel_spec


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

SAMPLE_RATE = 48000


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


# returns [x_train, y_train, x_test, y_test, x_val, y_val];


def split_data(data: pd.DataFrame):
    random_state = int(random.random() * 10)
    X_train = data.sample(n=1147, random_state=random_state)
    X_train["data"] = X_train["path"].apply(lambda x: extract_mfcc(x))
    Y_train = X_train[["emotion"]]
    remaining_subset = data.drop(index=X_train.index)

    X_val = remaining_subset.sample(n=143, random_state=random_state)
    X_val["data"] = X_val["path"].apply(lambda x: extract_mfcc(x))
    Y_val = X_val[["emotion"]]
    remaining_subset = remaining_subset.drop(index=X_val.index)

    remaining_subset["data"] = remaining_subset["path"].apply(lambda x: extract_mfcc(x))
    Y_test = remaining_subset[["emotion"]]

    return X_train, Y_train, X_val, Y_val, remaining_subset, Y_test


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


def transform_data(data: pd.DataFrame):
    from sklearn.preprocessing import StandardScaler

    mel_data = []
    for _, item in data.iterrows():
        mel_spec = get_mel_spec(item["data"], sample_rate=SAMPLE_RATE)
        mel_data.append(mel_spec)

    scaler = StandardScaler()
    mel_data = np.expand_dims(np.stack(mel_data, axis=0), 1)
    b, c, h, w = mel_data.shape
    mel_data = np.reshape(mel_data, newshape=(b, -1))
    mel_data = scaler.fit_transform(mel_data)
    mel_data = np.reshape(mel_data, newshape=(b, c, h, w))

    return mel_data


EPOCHS = 15000
BATCH_SIZE = 32


def train(
    xtrain_data: pd.DataFrame,
    ytrain_data: np.ndarray,
    X_val: pd.DataFrame,
    Y_val: np.ndarray,
):
    DATASET_SIZE = xtrain_data.shape[0]

    model = EmotionLSTM(num_of_emotions=len(EMOTIONS))
    device = torch.device(get_device())
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
    )

    print("preparing...")

    train_step = make_train_step(model, loss_fnc=loss_function, optimizer=optimizer)
    validate = make_validate_fnc(model, loss_function)
    train_xdata = transform_data(xtrain_data)
    X_val = transform_data(X_val)
    losses = []
    val_losses = []

    print("running on", device)
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

    print("saving model...")

    save_model(model)


def save_model(model):
    model_path = os.path.join(os.getcwd(), "models")
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_path, "emotion_lstm.pt"))
    print("model saved")


def load_model():
    model_file_path = os.path.join(os.getcwd(), "models", "emotion_lstm.pt")
    model = EmotionLSTM(num_of_emotions=len(EMOTIONS))
    device = torch.device(get_device())
    model.load_state_dict(torch.load(model_file_path, map_location=device))
    return model


if __name__ == "__main__":
    data_path = os.getenv("DATA_PATH")
    if data_path is None:
        print("DATA_PATH must given")
    else:
        df = prep_data(data_path)
        X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(df)
        tokenized_y_train = tokenizer(Y_train)
        tokenized_y_val = tokenizer(Y_val)
        train(X_train, tokenized_y_train, X_val, tokenized_y_val)
