import streamlit as st
import requests
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch

st.title('Image Net Model :tiger:')

# ResNet model
model = models.resnet50(pretrained=True)
model.eval()

# ImageNet class labels
imagenet_class_labels = requests.get(
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
).json()

# Upload image
uploaded_file = st.file_uploader("Choose an image...")

# Number of predictions
num_top_predictions = st.slider(
    "SLIDE TO CREATE PREDICTIONS:arrow_right:", min_value=1, max_value=5, value=5
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    predict_button = st.button("Predict")

    if predict_button:
        # Preprocess 
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch

        # Predictions
        with st.spinner("Predicting..."):
            with torch.no_grad():
                output = model(input_batch)

        # Predicted class and probability scores
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_probs, top_indices = torch.topk(probabilities, num_top_predictions)

        # Display top N predictions with probabilities
        st.subheader(f"YOU HAVE ASKED {num_top_predictions} PREDICTIONS:")
        for i in range(num_top_predictions):
            predicted_class = imagenet_class_labels[top_indices[i].item()]
            probability = top_probs[i].item()
            st.write(f"{i+1}. Class: {predicted_class}, Probability: {probability:.2%}")
