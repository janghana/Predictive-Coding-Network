import torch
from PIL import Image
from torchvision import transforms

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load the image
input_image = Image.open('./data/pac-man/illusory_contour_1.png').convert("RGB")  # replace with your image file

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # You may need to resize depending on your model input size
    transforms.ToTensor()
])

# Transform the image
input_image = transform(input_image)

# Add an extra batch dimension
input_image = input_image.unsqueeze(0)

# Ensure the model and image are on the same device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
input_image = input_image.to(device)

# Perform inference
results = model(input_image)

# Use the .pandas().xyxy[0] property of the results to get a Pandas DataFrame
results_df = results.xyxy[0]

# Iterate over the rows of the DataFrame to print the results
for index, row in results_df.iterrows():
    # Get the confidence and class
    confidence = row['confidence']
    object_class = row['name']  # The name column holds the name of the detected class
    print(f"Detected {object_class} with confidence {confidence}")
