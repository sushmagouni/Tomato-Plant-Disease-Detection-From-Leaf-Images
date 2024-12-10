import pickle
import warnings
warnings.filterwarnings('ignore')
import os
from PIL import Image
from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model('tmt.keras')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == "POST":
        # Get the uploaded image file
        f = request.files['image']

        # Define the path to save the image temporarily
        basepath = os.path.dirname(__file__)  # Current directory
        filepath = os.path.join(basepath, 'uploads', f.filename)
        
        # Ensure 'uploads' directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the uploaded file
        f.save(filepath)

        # Open and preprocess the image
        image = Image.open(filepath).resize((256, 256))
        image = np.asarray(image) / 255.0  # Normalize
        image = image.reshape(-1, 256, 256, 3)  # Reshape for the model input

        # Predict
        pred = np.argmax(model.predict(image))
        
        # Define class labels
        classes = [
            'Tomato_Bacterial_spot', 'TomatoEarly_blight', 'Tomatohealthy', 
            'TomatoLate_blight', 'TomatoLeaf_Mold', 'TomatoSeptoria_leaf_spot', 
            'TomatoSpider_mites_Two-spotted_spider_mite', 'TomatoTarget_Spot', 
            'TomatoTomato_mosaic_virus', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus'
        ]
        
        # Get the prediction label
        prediction = classes[pred]

        # Define custom messages based on the prediction
        if prediction == 'Tomato_Bacterial_spot':
            prediction_text = "The plant is predicted to have Bacterial spots."
        elif prediction == 'TomatoEarly_blight':
            prediction_text = "The plant is predicted to have Early Blights."
        elif prediction == 'Tomatohealthy':
            prediction_text = "The plant is predicted to be healthy."
        elif prediction == 'TomatoLate_blight':
            prediction_text = "The plant is predicted to have Late Blights."
        elif prediction == 'TomatoSeptoria_leaf_spot':
            prediction_text = "The plant is predicted to have Septoria Leaf Spot."
        elif prediction == 'TomatoSpider_mites_Two-spotted_spider_mite':
            prediction_text = "The plant is predicted to have Spider Mites."
        elif prediction == 'TomatoTarget_Spot':
            prediction_text = "The plant is predicted to have Target spots."
        elif prediction == 'TomatoTomato_mosaic_virus':
            prediction_text = "The plant is predicted to have Tomato Mosaic Virus."
        elif prediction == 'TomatoLeaf_Mold':
            prediction_text = "The plant is predicted to have Leaf Molds."
        elif prediction == 'Tomato_Tomato_Yellow_Leaf_Curl_Virus':
            prediction_text = "The plant is predicted to have Tomato Yellow Leaf Curl Virus."
        else:
            prediction_text = "Prediction could not be determined."

        # Render the result template with prediction text
        return render_template('results.html', prediction_text=prediction_text)

    # Return a default response if the method is not POST
    return render_template('predict.html')

# Run the app
if __name__ == "__main__":
    app.run(debug=True)