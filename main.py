#!/usr/bin/env python3

import gradio as gr
from typing import List
import torch

from utils import Utils, EMOTIONS, EMOTION_INTENSITY
from model import loss_function, EmotionLSTM
from train import Trainer

def trainer(epochs: int, batch_size: int, lr: float, data_path: List[str]):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = EmotionLSTM(num_of_emotions=len(EMOTIONS))
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
    )

    trainer = Trainer(model=model, optimizer=optimizer);
    trainer.prep_data(file_path=data_path)
    trainer.train(epochs=epochs, batch_size=batch_size, device=device);

    return "done";



demo = gr.Interface(
    gradio_trainer,
    [
        gr.Slider(128, 3600, value=512, label="Epoch Size", info="How many epoch for traning process"),
        gr.Slider(32, 64, value=32, label="Batch Size", info="Batch size for training process"),
        gr.Slider(0, 1, value=0.01, label="Learning Rate", info="Learning Rate for traning process"),
        gr.Textbox(
            label="data source path",
            info="/home/document/example_data_path",
            lines=3,
        ),
    ],
    "text",
    title="Emotion LSTM",
)

if __name__ == "__main__":
    demo.launch()
