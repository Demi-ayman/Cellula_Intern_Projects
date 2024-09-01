#class_names = ['CaS','CoS','Gum','MC','OC','OLP','OT']
#image =st.text_input('Enter Image name','Validation\\CaS\\a_80_0_1196.jpg')
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('teeth_class_VGG16.h5')
  # Replace with your model's path

# Function to preprocess the image for prediction
def preprocess_image(image):
    # Resize the image to match the model's input size
    img = image.resize((224, 224))  # Adjust dimensions as needed

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Normalize the image
    img_array = img_array / 255.0

    # Add an extra dimension to represent the batch size
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# Streamlit app
def main():
    st.title("Image Classification")

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...")

    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make a prediction
        prediction = model.predict(preprocessed_image) 


        # Get the predicted class
        predicted_class = np.argmax(prediction)
        class_names = ['CaS','CoS','Gum','MC','OC','OLP','OT']
        string = "This image most likely is: "+ class_names[np.argmax(predicted_class)]
        # Display the image and prediction
        st.image(image, caption='Uploaded Image')
        st.write("Predicted Class:" + string)

if __name__ == '__main__':
    main()