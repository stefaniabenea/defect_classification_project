import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import random
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_transforms(train=False):
    if train:
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.ColorJitter(brightness=0.2, contrast=0.2),
                                        transforms.RandomRotation(10),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),(0.5,0.5,0.5))]) 
        return transform
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,),(0.5,))
                                        ]) 
        return transform
    

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8,6))
    disp.plot(ax=ax, cmap="Greens", xticks_rotation=45)
    ax.set_title("Confusion matrix on test set")
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path,"confusion_matrix.png"))
    plt.show()


def get_albumentations_transforms(train=False):
    if train: 
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10,p=0.5),
            A.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
            ToTensorV2()
        ])
        return transform
    else:
        transform = A.Compose([
            A.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
            A.ToTensorV2()
        ])
        return transform
    
class AlbumentationsImageFolder(ImageFolder):
    def __init__(self, root, transform = None):
        super().__init__(root)
        self.alb_transformation = transform
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")
        image = np.array(image)
        if self.alb_transformation:
            augmented = self.alb_transformation(image=image)
            image = augmented['image']
        return image, target
    
def visualize_filters(model, layer_name, cols=4):
    all_layers = model.named_modules()
    all_layers = dict(all_layers)
    layer = all_layers.get(layer_name)

    if layer is None:
        print(f"Layer {layer_name} does not exist in the model.")
        return
    
    if not hasattr(layer, 'weight'):
        print(f"Layer {layer_name} does not have weight parameters.")
        return
    #(output_channels, input_channels, height_filer, width_filter)
    filters = layer.weight.data.cpu()
    num_filters = filters.shape[0]
    rows = (num_filters + cols - 1) // cols
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
    axs = axs.flatten()
    
    for i in range(num_filters):
        filt = filters[i]
        if filt.shape[0] > 1:
            filt_img = filt.mean(dim=0)
        else:
            filt_img = filt[0]
        
        axs[i].imshow(filt_img, cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(f"Filter {i}")
    
    for j in range(num_filters, len(axs)):
        axs[j].axis('off')
    
    plt.suptitle(f"Filters of layer '{layer_name}'", fontsize=16)
    plt.tight_layout()
    plt.show()

def visualize_augmentation(img_path, transform=None):
    image = Image.open(img_path).convert("RGB")
    image_np = np.array(image)

    if transform is None:
        transform = get_albumentations_transforms(train=True)

    augmented = transform(image=image_np)
    # after transform --> tensor 
    aug_img = augmented['image'].permute(1, 2, 0).numpy() 

    # de-normalize
    aug_img = aug_img * 0.5 + 0.5
    aug_img = np.clip(aug_img, 0, 1)

    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    axs[0].imshow(image_np)
    axs[0].set_title("Original")
    axs[0].axis('off')

    axs[1].imshow(aug_img)
    axs[1].set_title("Augmented")
    axs[1].axis('off')

    plt.show()

def visualize_activations(model, layer_name, img_path, transform, device ='cpu', cols=4):
    image = Image.open(img_path).convert("RGB")
    image = np.array(image)
    input_tensor = transform(image=image)['image'].to(device)
    
    activations = {}
    def hook_fn(module, input, output):
        activations['output']=output.detach().cpu()
    
    layer = dict(model.named_modules()).get(layer_name)
    if layer is None:
        print(f"Layer '{layer_name}' does not exist in the model.")
        return
    hook = layer.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        model(input_tensor.unsqueeze(0))
    hook.remove()

    # (1, num_filters, H,W)
    acts = activations['output']
    num_filters = acts.shape[1]
    rows = (num_filters + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axs = axs.flatten()

    for i in range(num_filters):
        act_img = acts[0, i].numpy()
        axs[i].imshow(act_img, cmap='viridis')
        axs[i].axis('off')
        axs[i].set_title(f"Filter {i}")

    for j in range(num_filters, len(axs)):
        axs[j].axis('off')

    plt.suptitle(f"Activations of layer '{layer_name}'", fontsize=16)
    plt.tight_layout()
    plt.show()


    

    



 