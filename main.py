import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from PIL import Image
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc

user_id=1

st.title("Potential Client Prediction Model")

# File Upload
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:

    # Load the data from the saved file
    data = pd.read_csv(uploaded_file)


    # Define the target behavior (interested in product)
    data["Interested"] = data["ROI"] > 160
    # Dropdown for checking missing values
    st.sidebar.header("Check Missing Values")
    check_missing_values = st.sidebar.checkbox("Check Missing Values")
    # If user selects to check missing values
    if check_missing_values:
        # Check for missing values
        missing_values = data.isnull().sum()
        st.write("Missing Values:", missing_values)

    # Data Preprocessing: Convert categorical variables to numerical using one-hot encoding
    data_encoded = pd.get_dummies(data, columns=["Job Title", "Experience Level", "Industry", "Company Size", "Recent Activity", "Budget Range"])

    # Select features for modeling
    features = ["ROI", "CS Team Size", "Customer Satisfaction", "Rating"] + list(data_encoded.columns[14:])  # Features from one-hot encoding

    # Split the data into features and target
    X = data_encoded[features]
    y = data_encoded["Interested"]

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression(max_iter=1000)  # Increased max_iter
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)  # Ensure output_dict=True

    # Display results
    st.subheader("Model Evaluation:")
    st.write("Accuracy:", accuracy)

    # Display classification report in a table
    report_df = pd.DataFrame.from_dict(report)
    st.text("Classification Report:")
    st.table(report_df)


    # Get the absolute coefficients as feature importance
    feature_importance = abs(model.coef_[0])

    # Create a DataFrame to display feature importance
    importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importance})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    # Print the feature importance
    st.header("Feature Importance:")
    st.dataframe(importance_df)

                    
        
    # Dropdown for visualizations
    st.sidebar.header("Visualizations")
    selected_visualization = st.sidebar.selectbox("Select Visualization", 
                                                ["Companies with Potential Clients","Relation of Interested",
                                                "Confusion Matrix Heatmap","Precision-Recall Curve",
                                                "ROC Curve",
                                                "Distribution of ROI for Interested Clients",
                                                "Distribution of ROI for Non-Interested Clients"
                                                    ])
    #prints company names
    if selected_visualization == "Companies with Potential Clients":
        st.header("Companies with Potential Clients:")
        # Assuming 'importance_df' contains feature importance
        potential_clients = importance_df[importance_df['Importance'] > 0.2]
        potential_client_features = potential_clients['Feature'].tolist()
        # Get companies with at least one important feature
        companies_with_potential_clients = data_encoded[data_encoded[potential_client_features].sum(axis=1) > 0]
        predicted_companies = companies_with_potential_clients['Company Name'].tolist()

        st.write(predicted_companies)


    # If user selects pairplot
    elif selected_visualization == "Relation of Interested":
        # Pairplot
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.subheader("Relation of Interested ")
        sns.pairplot(data, hue="Interested")
        st.pyplot()


    # If user selects Confusion Matrix Heatmap
    elif selected_visualization == "Confusion Matrix Heatmap":
        conf_matrix = confusion_matrix(y_test, y_pred)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.subheader("Confusion Matrix Heatmap")
        plt.figure(figsize=(12, 6))
        sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix Heatmap")
        st.pyplot()

    # If user selects Precision-Recall Curve
    elif selected_visualization == "Precision-Recall Curve":
        precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
        st.subheader("Precision-Recall Curve")
        plt.figure(figsize=(12, 6))
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        #st.pyplot()
        image = Image.open('C:\\Users\\SHYNI\\TRAINN\\pr.png')
        st.image(image)

    # If user selects ROC Curve
    elif selected_visualization == "ROC Curve":
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        st.subheader("ROC Curve")
        plt.figure(figsize=(12, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        st.pyplot()

    # If user selects Distribution of ROI for Interested Clients
    elif selected_visualization == "Distribution of ROI for Interested Clients":
        st.subheader("Distribution of ROI for Interested Clients")
        plt.figure(figsize=(12, 6))
        sns.histplot(data=data[data["Interested"]], x="ROI", bins=20, color="blue", label="Interested")
        plt.xlabel("ROI")
        plt.ylabel("Count")
        plt.title("Distribution of ROI for Interested Clients")
        plt.legend()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    # If user selects Distribution of ROI for Non-Interested Clients
    elif selected_visualization == "Distribution of ROI for Non-Interested Clients":
        st.subheader("Distribution of ROI for Non-Interested Clients")
        plt.figure(figsize=(12, 6))
        sns.histplot(data=data[~data["Interested"]], x="ROI", bins=20, color="orange", label="Not Interested")
        plt.xlabel("ROI")
        plt.ylabel("Count")
        plt.title("Distribution of ROI for Non-Interested Clients")
        plt.legend()
        st.pyplot()



