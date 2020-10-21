import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
import joblib
from sklearn import metrics, model_selection, preprocessing

import dataset
from models import RotNetclassifier
import config
import engine


def decode_preds(preds, encoder):
    preds = torch.softmax(preds, 1)
    preds = torch.argmax(preds, 1)
    preds = preds.detach().cpu().numpy()
    preds = encoder.inverse_transform(preds)
    return preds


def run_training():
    image_files = glob.glob("../input/dl_captcha/train_images/*.png")
    print(image_files.__len__())

    targets = [int(i.split("/")[-1].split(".")[0].split("_")[1])
               for i in image_files]
    print(targets.__len__())

    lbl_encoder = preprocessing.LabelEncoder()
    lbl_encoder.fit(targets)

    enc_targets = np.array(lbl_encoder.transform(targets))
    print(len(lbl_encoder.classes_))

    train_imgs, val_imgs, train_enc_targets, val_enc_targets, train_orig_targets, val_orig_targets = model_selection.train_test_split(
        image_files, enc_targets, targets, test_size=0.1, random_state=42)

    train_dataset = dataset.RotnetDataset(
        train_imgs, train_enc_targets, resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    
    print(len(train_dataset))

    val_dataset = dataset.RotnetDataset(
        val_imgs, val_enc_targets, resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))

    print(len(val_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=config.NUM_WORKERS
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS
    )

    model = RotNetclassifier(num_classes=len(lbl_encoder.classes_))
    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )

    for epoch in range(config.EPOCHS):
        train_loss = engine.train(model, train_loader, optimizer)
        valid_preds, valid_loss = engine.eval(model, val_loader)
        final_val_preds = []
        for vp in valid_preds:
            current_preds = decode_preds(vp, lbl_encoder)
            final_val_preds.extend(current_preds)

        combined = list(zip(val_orig_targets, final_val_preds))
        print(combined[:20])
        accuracy = metrics.accuracy_score(final_val_preds, val_orig_targets)
        print(
            f"Epoch={epoch}, Train Loss={train_loss}, Test Loss={valid_loss} Accuracy={accuracy}"
        )
        scheduler.step(valid_loss)
        joblib.dump(lbl_encoder, "../input/pickles/lbl_encoder.pkl")
        torch.save(model.state_dict(), "../input/pickles/captcha.pth")


if __name__ == "__main__":
    run_training()


run_training()
