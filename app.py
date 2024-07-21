import streamlit as st
import os
from dotenv import load_dotenv
import streamlit_google_oauth as oauth
import requests
from SimpleCalc import SimpleCalculator
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch

# Custom CSS for the navigation bar
custom_css = """
<style>
    body {
        margin: 0;
        padding: 0;
        display: flex;
        flex-direction: row;
        align-items: stretch;
        justify-content: flex-start;
        height: 100vh;
    }
    .navbar {
        background-color: #333;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 0;
        width: 100%;
    }
    .navbar span {
        color: white;
        text-align: center;
        padding: 10px 20px;
        transition: color 0.3s;
    }
    .navbar span:hover {
        color: #B6D0E2;
    }
    .navbar span.right {
        margin-left: auto;
    }
    
    .stApp {
        flex: 1; /* Take remaining width of the viewport */
        padding: 20px; /* Add some padding to the content area */
        display: flex;
        flex-direction: column;
        align-items: center; /* Center horizontally */
        justify-content: center; /* Center vertically */
    }
    
    .sidebar {
        width: 200px; /* Sidebar width */
        background-color: #333;
        padding: 20px;
        height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        z-index: 100;
    }
    
    .sidebar-title {
        color: white;
        font-size: 24px;
        margin-bottom: 20px;
    }
    
    .sidebar-link {
        color: white;
        padding: 10px 0;
        text-decoration: none;
        transition: color 0.3s;
    }
    
    .sidebar-link:hover {
        color: #B6D0E2;
    }


</style>

"""


# Render the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Load environment variables
load_dotenv('.env')
client_id = os.environ["CLIENT_ID"]
client_secret = os.environ["CLIENT_SECRET"]
redirect_uri = os.environ["REDIRECT_URI"]

# For User's login status
user_logged_in = False

if __name__ == "__main__":
    app_name = "MODEL"
    app_desc = "An ML-APP that authenticates users by Google Open-Authorization."
# User authentication
    login_info = oauth.login(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        app_name=app_name,
        app_desc=app_desc,
        logout_button_text="Logout",
    )
    
    if login_info:
        user_id, user_email = login_info
        user_logged_in = True

    # Create a navigation bar using custom HTML
    if user_logged_in:
         st.markdown(f'''
            <div class="navbar">
                     <span>Welcome to SANSTEN AI LABS!</span>
                <span>Hello, {user_email}</span>
                
            
                
            </div>
         ''', unsafe_allow_html=True)
    

    if user_logged_in:
        st.sidebar.title("MODELS")
        selected_model = st.sidebar.selectbox("Select a model", ["IMAGENET", "Calculator"])

        if selected_model == "IMAGENET":
            # ResNet model and ImageNet functionality
            st.title('IMAGENET Model :tiger:')
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

            

        elif selected_model == "Calculator":
            # Other model functionality
            st.title('SIMPLE CALCULATOR')
            calculator = SimpleCalculator()
            # Input fields
            num1_str = st.text_input('Enter the first number:')
            num2_str = st.text_input('Enter the second number:')

            # input str to numeric- floats here
            try:
                num1 = float(num1_str)
                num2 = float(num2_str)
            except ValueError:
                st.write("Please Enter Valid Numeric Values Only!")
            
             # Choice to user
                operation = st.selectbox("Select operation:", ("Add", "Subtract", "Multiply", "Divide"))

            if st.button('Calculate'):
                if operation == "Add":
                    result = calculator.add(num1, num2)
                elif operation == "Subtract":
                    result = calculator.subtract(num1, num2)
                elif operation == "Multiply":
                    result = calculator.multiply(num1, num2)
                elif operation == "Divide":
                    result = calculator.divide(num1, num2)
 
                st.write(f"Result: {result}")
 



