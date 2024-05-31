#!/usr/bin/env python3

from encoder import encode
from model import EmotionLSTM
from utils import EMOTIONS, extract_mfcc, get_device
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


def test(test_path: str):
    df = prep_data(test_path)
    df["data"] = df["path"].apply(lambda x: extract_mfcc(file_path=x, sr=48000))

    model = load_model("cpu")

    # model'i eval moda olmali
    model.eval()

    validate = make_validate_fnc(model, loss_function)
    X_test_tensor = torch.tensor(transform_data(df), device="cpu", dtype=torch.float)
    Y_test_tensor = torch.tensor(
        encode(df[["emotion"]]), dtype=torch.long, device="cpu"
    )

    test_loss, test_acc, _ = validate(X_test_tensor, Y_test_tensor)
    return test_loss, test_acc


if __name__ == "__main__":
    test_path = os.getenv("TEST_PATH")
    test_loss, test_acc = test(test_path)
    print(f"Test loss is {test_loss:.3f}")
    print(f"Test accuracy is {test_acc:.2f}%")
