from .utils import find_last_conv_layer
from .prediction import predict_and_visualize, visualize_all_classes_from_image
from keras import models
import matplotlib.pyplot as plt
import kagglehub

# dataset_path = kagglehub.dataset_download(
#     "paultimothymooney/chest-xray-pneumonia"
# )


dataset_path = kagglehub.dataset_download(
    "tawsifurrahman/covid19-radiography-database"
)

print(dataset_path)

from pathlib import Path

dataset_path = Path(dataset_path)

print(dataset_path)

print(dataset_path)

import random

root = dataset_path / "COVID-19_Radiography_Dataset"

covid_imgs = list(
    (root / "COVID" / "images").glob("*.png")
)

lung_opacity_imgs = list(
    (root / "Lung_Opacity" / "images").glob("*.png")
)

normal_imgs = list(
    (root / "Normal" / "images").glob("*.png")
)

viral_imgs = list(
    (root / "Viral Pneumonia" / "images").glob("*.png")
)

samples = [
    random.choice(covid_imgs),
    random.choice(lung_opacity_imgs),
    random.choice(normal_imgs),
    random.choice(viral_imgs),
]


class_names_bin = ["Normal", "Pneumonia"]
# class_names = class_names_bin
class_names = [
    "COVID",
    "Lung_Opacity",
    "Normal",
    "Viral Pneumonia"
]
model = models.load_model(
    r"D:\DATN\app\models\cov_masked_42.keras"
)

# last_conv_layer_name = find_last_conv_layer(model).name
last_conv_layer_name = "activation_3"
# last_conv_layer_name = "conv2d_3"
from pathlib import Path
import random

# test_dir = dataset_path / "chest_xray" / "test"

# normal_imgs = list(
#     (test_dir / "NORMAL").glob("*.jpeg")
# )

# pneumonia_imgs = list(
#     (test_dir / "PNEUMONIA").glob("*.jpeg")
# )

# normal_img = random.choice(normal_imgs)
# pneumonia_img = random.choice(pneumonia_imgs)

# # print(normal_img)
# # print(pneumonia_img)
# for img_path in [normal_img, pneumonia_img]:

#     fig = visualize_all_classes_from_image(
#         model=model,
#         image=str(img_path),
#         class_names=class_names,
#         true_class=img_path.parent.name,
#         method="saliency",
#         last_conv_layer_name=last_conv_layer_name,
#     )

#     plt.show()

for img_path in samples:

    fig = visualize_all_classes_from_image(
        model=model,
        image=str(img_path),
        class_names=class_names,
        true_class=img_path.parent.parent.name,
        method="integrated",
        last_conv_layer_name=last_conv_layer_name,
    )

    plt.show()