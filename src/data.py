import glob
import pandas as pd
from PIL import Image
import numpy as np
import os

angles = np.arange(10, 370, 10)
print(angles)

data_type = {
    "train": "train_images",
    "test": "test_images"
}

resize = (140, 140)

type = data_type["train"]

angle_list = []
for i in glob.glob(f"../input/single_test/*.png"):
    file_name = i.split("/")[-1].split(".")[-2]
    img = Image.open(i)
    angle_choice = np.random.choice(angles)
    print(angle_choice)
    img = img.rotate(angle_choice, expand=True)
    img = img.resize(resize, Image.ANTIALIAS)
    img.save(i)
    # angle_list.append(angle_choice)
    # os.rename(i, f"../input/dl_captcha/{type}/" +
    #           file_name + "_" + str(angle_choice) + ".png")
