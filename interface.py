#!/usr/bin/env python3

import gradio as gr
from typing import List
import torch
import subprocess
import multiprocessing
import os
import signal

from utils import Utils, EMOTIONS, EMOTION_INTENSITY
from model import loss_function, EmotionLSTM
from train import Trainer

def train(epochs: int, batch_size: int, lr: float, data_path: str):
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

process: None

def run_training_process(epochs: int, batch_size: int, lr: float, data_path: str, progress=gr.Progress(track_tqdm=True))-> str:
    global process;
    process = multiprocessing.Process(target=train, args=(epochs, batch_size, lr, data_path))
    process.start()
    try:
        process.join()
    except KeyboardInterrupt:
        print("Training interrupted. Terminating process...")
        process.terminate()
        process.join()
    return "Training process finished or terminated."


def kill_training()-> str:
    global process;
    if process and process.is_alive():
        os.kill(process.pid, signal.SIGKILL)
        process.join()
        return "Training process killed."

with gr.Blocks() as demo:
    gr.Label(value="Emotion LSTM", show_label=False)

    with gr.Row():
        epoch_slider_inp = gr.Slider(128, 3600, value=512, label="Epoch Size", info="How many epoch for traning process")
        batch_slider_inp = gr.Slider(32, 64, value=32, label="Batch Size", info="Batch size for training process")

    with gr.Row():
        lr_slider_inp = gr.Slider(0, 1, value=0.01, label="Learning Rate", info="Learning Rate for traning process")
        ds_text_box_inp = gr.Textbox(label="data source path", info="/home/whateverfuckyournameis/example_data_path",lines=3)

    start_btn = gr.Button("Run")
    stop_btn = gr.Button("Stop")

    output = gr.Textbox(label=None, visible=False)
    progress_bar = gr.HTML(visible=False)

    start_btn.click(fn=run_training_process,
                    inputs=[epoch_slider_inp,batch_slider_inp, lr_slider_inp, ds_text_box_inp],
                    outputs=[output, progress_bar],
                    show_progress=True)

    start_btn.click(
        fn=lambda: gr.update(visible=True),
        outputs=progress_bar,
        queue=False,
    )

    stop_btn.click(fn=kill_training, inputs=None, outputs=None);

if __name__ == "__main__":
    demo.launch()
