import pandas as pd
import numpy as np

def tokenizer(data: pd.DataFrame):
    emotions = {
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
        lambda x: 0 if emotions[x] == 8 else emotions[x]
    )
    return np.array(data.to_numpy().reshape((data.shape[0],)), dtype=np.int8)
