import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from urllib.request import urlopen
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Load the pretrained VGG16 model
model = models.vgg16(pretrained=True)
model.eval()

# Preprocessing transforms for input images
preprocess = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Get the list of image files in the directory
image_dir = "../data/pac-man/test"
image_files = os.listdir(image_dir)

# Iterate over the image files and perform classification
for image_file in image_files:
    # Load the image
    image_path = os.path.join(image_dir, image_file)
    input_image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_batch = input_batch.to(device)
    model.to(device)

    with torch.no_grad():
        output = model(input_batch)

    # Convert output to probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Load ImageNet class labels
    labels_url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
    labels = json.load(urlopen(labels_url))

    # Get the top 5 classes
    _, indices = torch.topk(probabilities, 5)
    probs_top5 = probabilities[indices].cpu().numpy()
    labels_top5 = [labels[idx] for idx in indices.cpu().numpy()]

    # Create the table data for the current image
    table_data = [['Class', 'Probability']]
    for label, prob in zip(labels_top5, probs_top5):
        table_data.append([label, prob])

    # Create the table plot
    plt.figure(figsize=(8, 6))
    table = plt.table(cellText=table_data, colWidths=[0.5, 0.5], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 1.5)
    plt.axis('off')
    plt.title(f"Top 5 predicted classes - {image_file}")
    plt.show()

    # Show the image
    plt.imshow(input_image)
    plt.title(f"Input Image - {image_file}")
    plt.axis('off')
    plt.show()
