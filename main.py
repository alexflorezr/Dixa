from flask import Flask, redirect, url_for, request, render_template

# Import Keras dependencies
from tensorflow.keras.models import load_model

# Import other dependecies
import numpy as np
import h5py
import os

#::: Flask App Engine :::
# Define a Flask app
app = Flask(__name__)

# ::: Prepare Keras Model :::
# Model files
encoder = load_model('./models/encoder.hdf5')
decoder = load_model('./models/decoder.hdf5')


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict/', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        #get form data
        encoder_input = request.form.get('encoder-input')
        #call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(encoder_input)
            #pass prediction to template
            return render_template('predict.html', prediction = prediction)
        except ValueError:
            return "Please Enter valid values"
        pass        
    pass

def preprocessDataAndPredict(encoder_input):
    #put all inputs in array    
    test_data = [float(x) for x in encoder_input if x != ',']
    #convert value data into numpy array and reshape
    test_data = np.array(test_data).astype(np.float).reshape(1,64)
    states_value = encoder.predict(test_data)
    decoder_token = np.zeros((1, 64))
    output_tokens, h1, c1, h2, c2 = decoder.predict([decoder_token] + states_value)
    prediction = []
    for i,a in enumerate(output_tokens[0,:,:]):
        prediction.append(np.argmax(output_tokens[0,i, :]))
    return prediction
    pass

if __name__ == '__main__':
    app.run(debug=True)