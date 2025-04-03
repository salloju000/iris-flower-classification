import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from model import get_model_info

# Load the trained model
mymodel = joblib.load("sid_model.pkl")

# Set page title and layout
st.set_page_config(page_title="Iris Flower Classification", layout="wide")

# Custom CSS for improved UI
st.markdown(
    """
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f4f4f9;
    }
    .title {
        text-align: center;
        color: #2e3d49;
        font-size: 36px;
        font-weight: bold;
    }
    .subheader {
        color: #3e6e8f;
        font-size: 24px;
    }
    .content {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .sidebar {
        background-color: #2e3d49;
        color: #ffffff;
    }
    .sidebar .stSlider {
        background-color: #d0e4e7;
    }
    .predicted-class {
        font-size: 24px;
        font-weight: bold;
    }
    .prediction-result {
        font-size: 22px;
        font-weight: bold;
        color: #4caf50;
    }
    .confusion-matrix {
        margin-top: 20px;
        margin-bottom: 40px;
    }
    .flower-image {
        width: 200px;
        height: 200px;
        object-fit: cover;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .feature-diagram {
        width: 300px;
        height: 300px;
        object-fit: cover;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add some spacing for better readability
st.markdown('<p class="title">Iris Flower Classification App</p>',
            unsafe_allow_html=True)

st.markdown(
    """
    <div class="content">
    <p style="text-align: justify;">
    This app uses the K-Nearest Neighbors (KNN) algorithm to classify Iris flowers based on their measurements.
    Select values for Sepal Length, Sepal Width, Petal Length, and Petal Width using the sliders on the left.
    The app will predict the species of the flower.
    </p>
    </div>
    """,
    unsafe_allow_html=True
)

# # Add an image or an icon for the iris flowers
# st.image("https://upload.wikimedia.org/wikipedia/commons/0/00/Iris_flowers.jpg", width=700)

# Add a diagram of the Sepal and Petal features (choose one method below)

# Option 1: Using an external URL for the diagram
st.image("https://miro.medium.com/v2/resize:fit:720/1*YYiQed4kj_EZ2qfg_imDWA.png",
         width=600, caption="Diagram of Iris with Sepal and Petal measurements")


# Section to explain the dataset
st.markdown('<p class="subheader">About the Iris Dataset</p>',
            unsafe_allow_html=True)
st.markdown(
    """
    The Iris dataset is a classic dataset in machine learning. It consists of 150 samples from three species of Iris flowers:
    - **Setosa**
    - **Versicolor**
    - **Virginica**

    The dataset contains four features for each flower:
    - **Sepal Length** (cm)
    - **Sepal Width** (cm)
    - **Petal Length** (cm)
    - **Petal Width** (cm)

    These features are used to classify the flowers into one of the three species.
    """
)

# Sidebar Sliders for user input
st.sidebar.header("Input Flower Features")
sl = st.sidebar.slider("Sepal Length (cm)", 1.0, 10.0, 5.0)
sw = st.sidebar.slider("Sepal Width (cm)", 1.0, 10.0, 3.0)
pl = st.sidebar.slider("Petal Length (cm)", 1.0, 10.0, 4.0)
pw = st.sidebar.slider("Petal Width (cm)", 1.0, 10.0, 1.0)

# Prediction based on user input
prediction = mymodel.predict([[sl, sw, pl, pw]])[0]

# Displaying the prediction with colored text for better visualization
if prediction == 0:
    predicted_class = "Setosa"
    color = "green"
    flower_image = "https://user-images.githubusercontent.com/68370376/187458343-09c00884-3fa5-4906-a773-1239b21d6dbb.png"
    width = 150
elif prediction == 1:
    predicted_class = "Versicolor"
    color = "blue"
    flower_image = "https://static.wixstatic.com/media/4a85d7_2c6de5add74541d4897105ef5b4ecd46~mv2.png/v1/fill/w_405,h_402,al_c,lg_1,q_85,enc_avif,quality_auto/4a85d7_2c6de5add74541d4897105ef5b4ecd46~mv2.png"
    width = 150
else:
    predicted_class = "Virginica"
    color = "red"
    flower_image = "https://ars.els-cdn.com/content/image/3-s2.0-B9780128147610000034-f03-01-9780128147610.jpg"
    width = 150

# Show the predicted class image
st.image(flower_image, caption=predicted_class, use_column_width=True)

# Displaying the prediction result
st.markdown(
    f'<p class="prediction-result">Predicted Class: <span style="color:{color};">{predicted_class}</span></p>',
    unsafe_allow_html=True
)

# Explanation of the prediction
st.markdown(
    """
    The model uses the measurements of the flower's sepal and petal (length and width) to predict the species. 
    The K-Nearest Neighbors (KNN) algorithm compares the input flower's features with known flowers in the training data 
    and classifies it into the most similar species.
    """
)

# Display model evaluation metrics
acc, cm = get_model_info()
st.markdown(f"### Model Accuracy: **{acc * 100:.2f}%**")

# Display confusion matrix as a heatmap with reduced size
st.subheader("Confusion Matrix", anchor="cm")
fig, ax = plt.subplots(figsize=(6, 4))  # Reduced size of the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
            'Setosa', 'Versicolor', 'Virginica'], yticklabels=['Setosa', 'Versicolor', 'Virginica'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(fig)
