import torch
from PIL import Image
import os
from model import CNN
from utils import get_albumentations_transforms
import numpy as np
import pandas as pd
import argparse
from dataset import prepare_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
a = torch.load("models/cnn_model.pth", map_location=device)
model.load_state_dict(a)
transforms = get_albumentations_transforms(train=False)
_,_, class_names = prepare_data(data_dir="data")

def predict_image(image_path, model, device, transforms):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    augmented_image = transforms(image=image)
    image_tensor = augmented_image['image'].unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output,dim=1)
        return prediction.item()

def predict_folder(folder_path, model, device, transforms, class_names, csv_path="results.csv"):
    
    results = []
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('jpg','png'))]
    for image in images:
        predicted_class = predict_image(image, model, device, transforms)
        results.append({"image":image, "prediction":class_names[predicted_class]})
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to csv")
    return results

parser = argparse.ArgumentParser(description="Script for CNN model inference")
parser.add_argument("--input", type=str, required=True, help="The path to an image or a folder of images")
args = parser.parse_args()

input_path = args.input

if os.path.isdir(input_path):
    print(f"{input_path} is processed as a folder")
    results = predict_folder(input_path, model, device, transforms, class_names)
    for r in results:
        print(f"{r['image']} is classified as {r['prediction']}")
    
else:
    print(f"{input_path} is processed as image")
    prediction = predict_image(input_path, model, device, transforms)
    print(f"The image {input_path} is classified as {class_names[prediction]}")


        





