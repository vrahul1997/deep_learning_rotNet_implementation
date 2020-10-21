import os
from numpy.lib.shape_base import split
from sklearn import metrics
from sklearn import preprocessing, model_selection
import glob
import torch
import pandas as pd
import numpy as np
import joblib

import dataset
from models import RotNetclassifier
import config


def decode_preds(preds, encoder):
    preds = torch.softmax(preds, 1)
    preds = torch.argmax(preds, 1)
    preds = preds.detach().cpu().numpy()
    preds = encoder.inverse_transform(preds)
    return preds


def run_test():
    image_files = sorted(glob.glob("../input/single_test/*.png"))
    print(image_files[:5])

    test_dataset = dataset.RotnetDataset(
        img_paths=image_files,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
    )
    lbl_enc = joblib.load("../input/pickles/lbl_encoder.pkl")
    model = RotNetclassifier(len(lbl_enc.classes_))
    model.load_state_dict(torch.load("../input/pickles/rotnet.pth"))

    test_preds = []
    for data in test_loader:
        model.eval()
        batch_preds, _ = model(**data)
        test_preds.append(batch_preds)
    

    all_preds = []
    for test_data in test_preds:
        current_preds = decode_preds(test_data, lbl_enc)
        for i in current_preds:
            all_preds.append(i)
    print(all_preds)

if __name__ == "__main__":
    run_test()