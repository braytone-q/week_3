from flask import Flask, request, render_template_string
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import io

app = Flask(__name__)
model = load_model("mnist_model.h5")

HTML_FORM = '''
<!doctype html>
<title>MNIST Digit Classifier</title>
<h1>Upload a 28x28 grayscale digit image</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file accept="image/*">
  <input type=submit value=Predict>
</form>
{% if prediction is not none %}
  <h2>Predicted Digit: {{ prediction }}</h2>
  <h3>Confidence: {{ confidence }}</h3>
  <img src="data:image/png;base64,{{ image_data }}" width="150"/>
{% endif %}
'''

import base64

def preprocess_image(image):
    image = image.convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array, image

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_data = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            image = Image.open(file.stream)
            arr, img = preprocess_image(image)
            pred = model.predict(arr)
            predicted_digit = int(np.argmax(pred))
            conf = float(np.max(pred))
            # Convert image to base64 for display
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            prediction = predicted_digit
            confidence = f"{conf:.2f}"
            image_data = img_b64
    return render_template_string(HTML_FORM, prediction=prediction, confidence=confidence, image_data=image_data)

if __name__ == '__main__':
    app.run(debug=True)
