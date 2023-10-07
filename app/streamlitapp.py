
import streamlit as st
import os
import imageio

import tensorflow as tf
from utils import load_alignments, num_to_char,load_data, load_video
from modelutil import load_model,train_model, load_model_lip,predict_audio_model

with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')


st.title('LipNet Full Stack App') 
# Generating a list of options or videos 
options = os.listdir(os.path.join('..', 'Final_test_dataset', 'Final_test_dataset'))
selected_video = st.selectbox('Choose video', options)

def get_annotations(file_path):
    file_name = os.path.basename(file_path).split(".")[0]
    with open("../Final_test_dataset/align.csv", "r") as f:
        lines = f.readlines()
    
    for i in lines:
        if i.startswith(file_name):
            return i.split("|")[1]




col1, col2 ,col3 = st.columns([3,2,2])

if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..','Final_test_dataset','Final_test_dataset', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        video = load_video(file_path)
        annotation = get_annotations(file_path)
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width=200) 

        # st.info('This is the output of the machine learning model as tokens')
       
    
    with col3:
        st.info("This is the original text:")
        characters = annotation

        # Convert the characters tensor to a Python string
        text = tf.strings.reduce_join(characters).numpy().decode('utf-8')

        # Print the resulting text
     
        st.text(text)
        model = load_model_lip()
        
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        # st.text(decoder)

        # Convert prediction to text
        st.info('Prediction By Visual Speech Recognition Model:')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        # text_placeholder = st.empty()
        st.info('Prediction By  Audio Speech Recognition Model:')
        audio_model = predict_audio_model(file_path)
        st.text(" ")
        st.text(str(audio_model))
            
        import streamlit as st

# Create a Streamlit button
        
        

       

        #
        # st.button("Prediction By ASR", type="primary")
      
       
        value_op = "bin blue at f two now"
 