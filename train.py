#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import torch
from model import EmotionLSTM, loss_function
from utils import  save_model, extract_mfcc, extract_mel_spectogram
import encoder

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
RANDOM_STATE = 42


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
            gender = "female" if int(path_prefix[6]) % 2 == 0 else "male"
            data.append([EMOTIONS[emotion], emotion_intense, gender, full_path])
    return pd.DataFrame(
        data,
        columns=["emotion", "emotion_intensity", "gender", "path"],
    )

def split_data(data: pd.DataFrame):
    X_train = data.sample(n=1150, random_state=RANDOM_STATE)

    X_train["data"] = X_train["path"].apply(lambda p: extract_mfcc(file_path=p, sr=SAMPLE_RATE))
    Y_train = X_train[["emotion"]]
    remaining_subset = data.drop(index=X_train.index)

    X_val = remaining_subset.sample(n=140, random_state=RANDOM_STATE)
    X_val["data"] = X_val["path"].apply(lambda p: extract_mfcc(file_path=p, sr=SAMPLE_RATE))
    Y_val = X_val[["emotion"]]
    remaining_subset = remaining_subset.drop(index=X_val.index)

    remaining_subset["data"] = remaining_subset["path"].apply(lambda p: extract_mfcc(file_path=p, sr=SAMPLE_RATE))
    Y_test = remaining_subset[["emotion"]]

    return X_train, Y_train, X_val, Y_val, remaining_subset, Y_test

   
def transform_data(data: pd.DataFrame):
    from sklearn.preprocessing import StandardScaler

    mel_data = []
    for _, item in data.iterrows():
        mel_data.append(extract_mel_spectogram(data=item["data"], sr=SAMPLE_RATE))

    scaler = StandardScaler()
    mel_data = np.expand_dims(np.stack(mel_data, axis=0), 1)
    print(mel_data.shape)
    b, c, h, w = mel_data.shape
    mel_data = np.reshape(mel_data, newshape=(b, -1))
    mel_data = scaler.fit_transform(mel_data)
    mel_data = np.reshape(mel_data, newshape=(b, c, h, w))

    return mel_data


def make_validate_fnc(model, loss_fnc):
    def validate(X, Y):
        with torch.no_grad():
            model.eval()
            output_logits, output_softmax = model(X)
            predictions = torch.argmax(output_softmax, dim=1)
            accuracy = torch.sum(Y == predictions) / float(len(Y))
            loss = loss_fnc(output_logits, Y)
        return loss.item(), accuracy * 100, predictions

    return validate


def make_train_step(model, loss_fnc, optimizer):
    def train_step(X, Y):
        model.train()
        output_logits, outpout_softmax = model(X)
        predictions = torch.argmax(outpout_softmax, dim=1)
        accuracy = torch.sum(Y == predictions) / float(len(Y))
        loss = loss_fnc(output_logits, Y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        return loss.item(), accuracy * 100

    return train_step

EPOCHS = 1500
BATCH_SIZE = 32

def train(
    xtrain_data: pd.DataFrame,
    ytrain_data: np.ndarray,
    X_val: pd.DataFrame,
    Y_val: np.ndarray,
):
    DATASET_SIZE = xtrain_data.shape[0]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = EmotionLSTM(num_of_emotions=len(EMOTIONS))
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
    )

    train_step = make_train_step(model, loss_fnc=loss_function, optimizer=optimizer)
    validate = make_validate_fnc(model, loss_function)
    train_xdata = transform_data(xtrain_data)
    X_val = transform_data(X_val)
    losses = []
    val_losses = []

    print("running")

    for epoch in range(EPOCHS):
        idx = np.random.permutation(DATASET_SIZE)
        train_xdata = train_xdata[idx, :, :, :]
        ytrain_data = ytrain_data[idx]

        epoch_acc = 0
        epoch_loss = 0

        iters = int(DATASET_SIZE / BATCH_SIZE)
        for i in range(iters):
            ## linear data frame icinde `BATCH_SIZE` stride seklinde atlayarak gider
            batch_start = i * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, DATASET_SIZE)
            ## data frame sonunda `BATCH_SIZE`'dan daha az data ise 
            ## last data almayi garantilemek icin konulmustur
            actual_batch_size = batch_end - batch_start

            X = train_xdata[batch_start:batch_end, :, :, :]
            Y = ytrain_data[batch_start:batch_end]

            X_tensor = torch.tensor(X, dtype=torch.float, device=device)
            Y_tensor = torch.tensor(Y, dtype=torch.long, device=device)

            loss, acc = train_step(X_tensor, Y_tensor)
            epoch_acc += acc * actual_batch_size / train_xdata.shape[0]
            epoch_loss += loss * actual_batch_size / train_xdata.shape[0]

        X_val_tensor = torch.tensor(X_val, device=device, dtype=torch.float)
        Y_val_tensor = torch.tensor(Y_val, device=device, dtype=torch.long)
        val_loss, val_acc, _ = validate(X_val_tensor, Y_val_tensor)
        losses.append(epoch_loss)
        val_losses.append(val_loss)
        print(
            f"Epoch {epoch} --> loss:{epoch_loss:.4f}, acc:{epoch_acc:.2f}%, val_loss:{val_loss:.4f}, val_acc:{val_acc:.2f}%"
        )

        # if epoch reaches 0.001 before than epoch iteracton it will break
        if epoch_loss <= 1e-3:
            break
    print("saving model")
    save_model(model)



if __name__ == "__main__":
    DATA_PATH = os.getenv("DATA_PATH")
    if DATA_PATH is None:
        print("DATA_PATH should be given")
    else:
        df = prep_data(DATA_PATH)
        X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(df)
        train(X_train, encoder.encode(Y_train), X_val, encoder.encode(Y_val))
