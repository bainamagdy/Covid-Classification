
import numpy as np  
import streamlit as st
import cv2 
from PIL import Image
from tensorflow.keras.models import load_model


# إعداد صفحة Streamlit
st.set_page_config(page_title="COVID-19 X-ray Classifier", page_icon="🦠", layout="centered")

st.title("🦠 COVID-19 X-ray Image Classifier")
st.markdown("""
This app uses a deep learning model to classify chest X-ray images as **Covid**, **Normal**, or **Pneumonia**.

- Upload a chest X-ray image (JPG, JPEG, PNG).
- Click **Predict** to see the result.
""")

# الشريط الجانبي
with st.sidebar:
    st.header("About")
    st.info("""
    - Model: Custom CNN (Keras)
    - Input size: 224x224x3
    - Classes: Covid, Normal, Pneumonia
    """)
    st.markdown("---")
    st.write("Developed by Baina magdy")

# رفع الصورة
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# دالة معالجة الصورة
def preprocessing(img_in, x_axis, y_axis, dim):
    img = np.array(img_in)  # تحويل الصورة إلى مصفوفة
    img = cv2.resize(img, (x_axis, y_axis))  # تغيير الحجم
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img / 255.0  # تطبيع القيم
    img = np.reshape(img, (1, x_axis, y_axis, dim))  # إضافة بعد batch
    return img

# تعريف المخرجات الممكنة
class_mapping = {
    0: 'Covid', 
    1: 'Normal',
    2: 'Pneumonia'
}

# دالة التنبؤ
def predict(img, model):
    predict = model.predict(img)
    class_index = np.argmax(predict, axis=1)
    class_label = class_mapping[class_index[0]]
    return class_label

# تنفيذ عند رفع صورة
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image')
    st.markdown("---")
    st.write("**Image details:**")
    st.write(f"Format: {img.format}")
    st.write(f"Size: {img.size}")
    st.write(f"Mode: {img.mode}")
    st.markdown("---")
    
    if st.button('🔍 Predict', use_container_width=True):
        up_img = preprocessing(img, 224, 224, 3)
        model = load_model("covid_19_model.h5")
        result = predict(up_img, model)
        st.success(f'The image is classified as: **{result}**')

else:
    st.info("Please upload a chest X-ray image to get started.")



