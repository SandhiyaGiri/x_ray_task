# Objective:
Implement Faster R-CNN model or choose an alternative architecture of your own choice for pathology detection in XRAY images. Evaluate and compare the model's performance in identifying various pathologies and develop a simple web application to showcase your findings.

## Dataset: 
The dataset contains 3300 images as: 
1.	600 images on - class A 
2.	600 images on - class B 
3.	600 images on - class C 
4.	1500 images on- class D
Out of this some of the images are corrupted. After removing corrupted files, the dataset has 3,254 images. To address imbalance, only 600 of the ~1,500 images in class D (normal-rays) are used for training.
The JSON file contains 4 dictionaries, one for each class, with image names as keys. The values are strings in the format: <class x_center y_center width height>. Since PyTorch is used in this project, the JSON file is converted to a format that matches the required target dataset parameters to train the model. The modified JSON file is a list of dictionaries with image_name as key and the parameters listed below as values. 

![image](https://github.com/user-attachments/assets/030a7db7-ffab-41f2-a1f4-3756976e3dcf)

Note: The bbox values are not changed

## 1.	Data Preprocessing:
The XRayDataset class is a custom PyTorch dataset that can handle image loading, annotation handling, and preprocessing - CLAHE. (Contrast Limited Adaptive Histogram Equalization -  used to improve the contrast of medical images making the details more visible.)
Arguments:
image_dir (str): Path to the directory containing X-ray images.
annotations (dict): A dictionary with image names as keys and annotations (bounding boxes and labels) as values.
transform (callable, optional): A function/transform to be applied to each sample (e.g., resizing, normalization).
apply_clahe_flag (bool, optional): If True, applies CLAHE to the images for contrast enhancement. Default is False.

Methods:
__len__(): Returns the number of images in the dataset.
__getitem__(idx): Loads an image and its corresponding annotations (bounding boxes and labels) at the given index.

Visualization to understand the bounding boxes:

The function “visualize_bounding_boxes”, takes normalized bounding box coordinates [x_center, y_center, width, height] and converts them into the format [x_min, y_min, x_max, y_max] by scaling them based on the image's width and height.

Example Output:

•	Bounding Box: The bounding box will be drawn around the detected object (e.g., a specific abnormality in an X-ray).

•	Label: The label for the object (e.g., 'A', 'B', etc.) will appear above the bounding box in red.

## 2.	Model Implementation:
Faster R-CNN for detecting pathologies in X-rays:

2.1. Backbone Network
Backbone Architecture: Utilizes ResNet-50 with Feature Pyramid Network (FPN) for effective feature extraction from input X-ray images.
Pre-trained Model: Loads a pre-trained Faster R-CNN model that has been trained on the COCO dataset, facilitating improved performance on the specific pathology detection task.

2.2. Region Proposal Network (RPN)
RPN Functionality: Automatically generates candidate object proposals from the feature maps, identifying regions that potentially contain pathologies.
Integration: The RPN is integrated into the Faster R-CNN architecture and is trained jointly with the rest of the model to improve localization and classification performance.

2.3. RoI Pooling
RoI Pooling Mechanism: Converts variable-sized proposals from the RPN into fixed-size feature maps for downstream classification and bounding box regression.
Internal Handling: The RoI pooling layer is handled within the pre-trained model, streamlining the workflow and reducing implementation complexity.

2.4. Custom Classifier
Modified Head: Replaces the box predictor of the pre-trained model to match the number of classes specific to this pathology detection task (e.g., 4 object classes + 1 background class).
Class Count: Configured to output predictions for 5 classes, ensuring that the model can accurately identify the presence of different pathologies.

2.5. Training Loop
Training Function: Implements a function (train_one_epoch) that iterates through the dataset, performing forward and backward passes, and updating model weights.
Loss Calculation: Combines classification and bounding box regression losses to optimize the model during training.
Progress Tracking: Prints training loss at regular intervals for monitoring the training process.

# 3. Web Application Development
The application leverages PyTorch and the torchvision library to perform object detection, allowing users to upload images and view the resulting predictions with bounding boxes and confidence scores.

Features

Image Upload: Users can upload X-ray images to be evaluated by the Faster R-CNN model.

Object Detection: The model detects various pathologies in the uploaded images and provides bounding box annotations.

Visualization: Predictions are visualized with bounding boxes drawn around detected objects, along with class labels and confidence scores displayed.
