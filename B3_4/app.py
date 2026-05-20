import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps

#config page
st.set_page_config (page_title='ASL',
                page_icon=':smiley:',
                layout='centered',
                initial_sidebar_state='auto'
                )

st.title("Nhan dien chu cai ASL")

#load model
@st.cache_resource
def load_model(model_url):
    model = keras.models.load_model('asl_model.h5')
    return model

model= load_model('asl_model.h5')

class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space', 'nothing'
]
IMG_SIZE = 64

# -------------------------
# ham ho tro (xu ly hinh anh input)
# -------------------------
def preprocess_image(pil_img, input_size=(64, 64)):
    pil_img = pil_img.convert("RGB")  # Convert về RGB (tránh lỗi với hình RGBA)
    img = pil_img.resize(
        input_size
    )  # Resize hình cho phù hợp với input size của mô hình
    img_array = image.img_to_array(
        img
    )  # Chuyển từ kiểu dữ liệu hình sang kiểu numpy array

    img_array = np.expand_dims(img_array, axis=0)  # Thêm n=1 để batch_size=1
    test_datagen = image.ImageDataGenerator(  # Bắt buộc áp dụng các phương pháp tiền xử lý như tập train
        samplewise_center=True, samplewise_std_normalization=True
    )
    img_generator = test_datagen.flow(
        img_array, batch_size=1
    )  # Thay vì sử dụng `flow_from_directory` thì chỉ sử dụng `flow`
    return img_generator

#tao dao dien

#chon kieu test
input_type= st.radio('chose a type:',('Upload Image','Use Camera'), index=0)

if input_type =="use camera":
    st.warning("Camera input is not supported in this demo. Please upload an image instead.")

#Khung input hinh
uploaded_file= st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

#Not found hinh anh upload
if input_type == 'Upload Image' and uploaded_file is None:
    st.warning("Image not found. Please upload an image to proceed.")

elif input_type=="Upload Image"and uploaded_file is not None:
    img=Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)



if st.button("Predict"):
    # loading
    with st.spinner("Predicting..."):
        img_input = preprocess_image(img)

        # du doan hinh anh
    predictions = model.predict(img_input)
    prediction_idx = np.argmax(predictions)
    predicted_label = class_names[prediction_idx]
    confidence = np.max(predictions)
    st.write(f"**Prediction:** {predicted_label} with {confidence*100:.2f}% confidence.")
