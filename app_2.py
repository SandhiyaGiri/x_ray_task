import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import numpy as np


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def get_faster_rcnn_model(num_classes):

    model = fasterrcnn_resnet50_fpn(pretrained=True)  # Load pre-trained Faster R-CNN

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def plot_predictions(image, outputs, threshold=0.6):
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')
    image_width, image_height = image.size
    for box, score, label in zip(outputs['boxes'], outputs['scores'], outputs['labels']):
        if score > threshold:  # Only plot if score is above the threshold
            # Unpack the box coordinates
            x_min, y_min, x_max, y_max = box.cpu().numpy()
            
            # Create a Rectangle patch
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # Display class label and score
            ax.text(x_min, y_min, f'Class: {label.item()}, Score: {score.item():.2f}',
                    color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
    
    plt.axis('off')
    plt.show()

def evaluate_and_plot(model, image_paths, device, threshold=0.6):

    model.eval()
    with torch.no_grad():
        images = [Image.open(path).convert("RGB") for path in image_paths]

        # Convert images to tensors and normalize

        images_tensor = [normalize(transforms.ToTensor()(image)).to(device) for image in images]
        # Get predictions

        outputs = model(images_tensor)  # Get predictions

        print(outputs)

        # Plot predictions for each image

        for image, output, path in zip(images, outputs, image_paths):

            plot_predictions(image, output, threshold)
            
image_paths = [r"D:\5C_Tasks\5C_interview\2021_09_27_F551AD05_9110A808_2B26D00F.jpeg"]# Provide paths to test images

num_classes = 4 + 1
model = get_faster_rcnn_model(num_classes)

model.eval()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)

evaluate_and_plot(model, image_paths, device)