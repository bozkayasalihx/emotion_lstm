#!/usr/bin/env python3

from encoder import encode
from model import EmotionLSTM
from utils import EMOTIONS, get_device
from model import loss_function
from train import prep_data, split_data, make_validate_fnc, transform_data
import torch
import os


def load_model(device: str):
    MODEL_PATH = os.path.join(os.getcwd(), "models")
    model = EmotionLSTM(len(EMOTIONS))
    model.load_state_dict(
        torch.load(
            os.path.join(MODEL_PATH, "emotion_lstm.pt"),
            map_location=torch.device(device),
        )
    )
    print("Model is loaded from {}".format(os.path.join(MODEL_PATH, "emotion_lstm.pt")))
    return model


if __name__ == "__main__":
    DATA_PATH = os.getenv("DATA_PATH")
    device = get_device()
    df = prep_data(DATA_PATH)
    splitedDf = split_data(df)

    X_test = splitedDf[0]
    Y_test = splitedDf[1]
    model = load_model(device)

    model.eval()
    model.to(device)
    validate = make_validate_fnc(model, loss_function)
    X_test_tensor = torch.tensor(
        transform_data(X_test), device=device, dtype=torch.float
    )
    Y_test_tensor = torch.tensor(encode(Y_test), dtype=torch.long, device=device)

    test_loss, test_acc, _ = validate(X_test_tensor, Y_test_tensor)
    print(f"Test loss is {test_loss:.3f}")
    print(f"Test accuracy is {test_acc:.2f}%")
