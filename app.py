import numpy as np
import pickle
import streamlit as st


loaded_model = pickle.load(open(
    'trained_model.sav',
    'rb'
))


def iris_prediction(input_data):
    # Convert input_data to numpy float array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float).reshape(1, -1)
    
    prediction = loaded_model.predict(input_data_as_numpy_array)

    if prediction[0] == 0:
        return 'Iris-virginica'
    elif prediction[0] == 1:
        return 'Iris-setosa'
    else:
        return 'Iris-versicolor'


def main():
    st.title('Iris Prediction Web App')
    SepalLengthCm = st.text_input('Sepal Length')
    SepalWidthCm = st.text_input('Sepal Width')
    PetalLengthCm = st.text_input('Petal Length')
    PetalWidthCm = st.text_input('Petal Width')

    prediction = ''
    if st.button('Predict'):
        try:
            # Convert inputs to float
            input_data = [float(SepalLengthCm), float(SepalWidthCm),
                          float(PetalLengthCm), float(PetalWidthCm)]
            
            prediction = iris_prediction(input_data)
        except ValueError:
            prediction = "❌ Please enter valid numeric values!"
    
    st.success(prediction)


if __name__ == "__main__":
    main()

