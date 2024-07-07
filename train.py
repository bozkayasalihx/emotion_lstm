#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from utils import Utils, EMOTIONS, EMOTION_INTENSITY
from model import loss_function, EmotionLSTM
import encoder



SAMPLE_RATE = 48000
RANDOM_STATE = 42

EPOCHS = 512
BATCH_SIZE = 32


class Trainer(Utils):
    def __init__(self, model: torch.Tensor, optimizer: callable):
        self.model = model
        self.optim = optimizer

    def prep_data(self, file_path: str):
        if file_path is None:
            return "file path must not be empty";

        data = []
        for dir in os.listdir(file_path):
            each_audio_path = os.path.join(file_path, dir)
            if os.path.isfile(each_audio_path):
                continue

            for audio_path in os.listdir(each_audio_path):
                full_path = os.path.join(each_audio_path, audio_path)
                path_prefix = audio_path.split(".")[0].split("-")
                if len(path_prefix) < 2:
                    continue;
                emotion = int(path_prefix[2])
                emotion_intense = EMOTION_INTENSITY[int(path_prefix[3])]
                if emotion == 8:
                    emotion = 0
                gender = "female" if int(path_prefix[6]) % 2 == 0 else "male"
                data.append([EMOTIONS[emotion], emotion_intense, gender, full_path])

        pd_data = pd.DataFrame(data,columns=["emotion", "emotion_intensity", "gender", "path"])

        self.X_train = pd_data.sample(n=1150, random_state=RANDOM_STATE);
        self.X_train["data"] = self.X_train["path"].apply(lambda p: self.extract_mfcc(file_path=p, sr=SAMPLE_RATE))
        self.Y_train = encoder.encode(self.X_train[["emotion"]])
        rsubset: pd.DataFrame  = pd_data.drop(index=self.X_train.index)

        self.X_val = rsubset.sample(n=140, random_state=RANDOM_STATE)
        self.X_val["data"] = self.X_val["path"].apply(lambda p: self.extract_mfcc(file_path=p, sr=SAMPLE_RATE));
        self.Y_val = encoder.encode(self.X_val[["emotion"]])

        self.X_test = rsubset.drop(index=self.X_val.index);
        self.X_test["data"] = self.X_test["path"].apply(lambda p: self.extract_mfcc(file_path=p, sr=SAMPLE_RATE));
        self.Y_test = encoder.encode(self.X_test[["emotion"]])


    def transform_data(self, data: pd.DataFrame):
        from sklearn.preprocessing import StandardScaler

        mel_data = []
        for _, item in data.iterrows():
            mel_data.append(self.extract_mel_spectogram(data=item["data"], sr=SAMPLE_RATE))

        scaler = StandardScaler()
        mel_data = np.expand_dims(np.stack(mel_data, axis=0), 1)
        b, c, h, w = mel_data.shape
        mel_data = np.reshape(mel_data, newshape=(b, -1))
        mel_data = scaler.fit_transform(mel_data)
        mel_data = np.reshape(mel_data, newshape=(b, c, h, w))

        return mel_data




    def make_validate_fnc(self, loss_fnc: callable):
        def validate(X, Y):
            with torch.no_grad():
                self.model.eval()
                output_logits, output_softmax = self.model(X)
                predictions = torch.argmax(output_softmax, dim=1)
                accuracy = torch.sum(Y == predictions) / float(len(Y))
                loss = loss_fnc(output_logits, Y)
            return loss.item(), accuracy * 100, predictions

        return validate


    def make_train_step(self, loss_fnc: callable, optimizer: callable):
        def train_step(X, Y):
            self.model.train()
            output_logits, outpout_softmax = self.model(X)
            predictions = torch.argmax(outpout_softmax, dim=1)
            accuracy = torch.sum(Y == predictions) / float(len(Y))
            loss = loss_fnc(output_logits, Y)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            return loss.item(), accuracy * 100

        return train_step



    def train(self, epochs=512, batch_size=32, device=str):

        DATASET_SIZE = self.X_train.shape[0]

        train_step = self.make_train_step(loss_fnc=loss_function, optimizer=self.optim)
        validate = self.make_validate_fnc(loss_fnc=loss_function)
        x_train = self.transform_data(self.X_train)
        X_val = self.transform_data(self.X_val)
        losses = []
        val_losses = []

        print("running")

        for epoch in range(epochs):
            idx = np.random.permutation(DATASET_SIZE)
            x_train = x_train[idx, :, :, :]
            ytrain_data = self.Y_train[idx];

            epoch_acc = 0
            epoch_loss = 0

            iters = int(DATASET_SIZE / batch_size)
            for i in tqdm(range(iters)):
                ## linear data frame icinde `BATCH_SIZE` stride seklinde atlayarak gider
                batch_start = i * batch_size
                batch_end = min(batch_start + batch_size, DATASET_SIZE)
                ## data frame sonunda `BATCH_SIZE`'dan daha az data ise
                ## last data almayi garantilemek icin konulmustur
                actual_batch_size = batch_end - batch_start

                X = x_train[batch_start:batch_end, :, :, :]
                Y = ytrain_data[batch_start:batch_end]

                X_tensor = torch.tensor(X, dtype=torch.float, device=torch.device(device))
                Y_tensor = torch.tensor(Y, dtype=torch.long, device=torch.device(device))

                loss, acc = train_step(X_tensor, Y_tensor)
                epoch_acc += acc * actual_batch_size / x_train.shape[0]
                epoch_loss += loss * actual_batch_size / x_train.shape[0]

            X_val_tensor = torch.tensor(X_val, device=torch.device(device), dtype=torch.float)
            Y_val_tensor = torch.tensor(self.Y_val, device=torch.device(device), dtype=torch.long)
            val_loss, val_acc, _ = validate(X_val_tensor, Y_val_tensor)
            losses.append(epoch_loss)
            val_losses.append(val_loss)
            print(
                f"Epoch {epoch} --> loss:{epoch_loss:.4f}, acc:{epoch_acc:.2f}%, val_loss:{val_loss:.4f}, val_acc:{val_acc:.2f}%"
            )

            # if epoch reaches 0.001 before than epoch iteracton it will break
            if epoch_loss <= 1e-3:
                break

        # print("saving model")
        # save_model(model)


if __name__ == "__main__":
    DATA_PATH = os.getenv("DATA_PATH")
    if DATA_PATH is None:
       print("DATA_PATH should be given")
    else:
        ### should take model and data path
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model = EmotionLSTM(num_of_emotions=len(EMOTIONS))
        model.to(device)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
        )

        trainer = Trainer(model=model, optimizer=optimizer);
        trainer.prep_data(file_path=DATA_PATH)
        trainer.train(epochs=512, batch_size=32, device=device)
