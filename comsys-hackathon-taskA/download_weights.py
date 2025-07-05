import os
import gdown


if not os.path.exists("weights/best_convnext_gender_model.pth"):
    url = "https://drive.google.com/uc?id=1Q97l7ROkH5MrO3hc-5D6ugAo2sOyXaMQ"
    gdown.download(url, "weights/best_convnext_gender_model.pth", quiet=False)

