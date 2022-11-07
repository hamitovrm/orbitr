import io
import torch
import requests
import streamlit as st
from PIL import Image
import json



API_URL_ta = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-mul"
headers = {"Authorization": f"Bearer {'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'}"}

def translate(payload, API_URL):
	response = requests.post(API_URL, headers=headers, json=payload )
	return response.json
	


API_URL_IMG = "https://api-inference.huggingface.co/models/ydshieh/vit-gpt2-coco-en-ckpts"
headers = {"Authorization": f"Bearer {'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'}"}

def img2txt(image):
    with io.BytesIO() as buf:
        image.save(buf, 'jpeg')
        image_bytes = buf.getvalue()
    response = requests.post(API_URL_IMG, headers=headers, data=image_bytes )
    return response.json


def predict(image):
    preds = img2txt(image)
    preds = [pred.strip() for pred in preds()]
    st.write(str(preds))
    return preds

def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        return image_data
    else:
        return None

def print_predictions(preds):
    for cl in preds:
        #st.write(str(cl).replace('_'," "))
        en_text=str(cl).replace('_'," ")
        trans_ta = translate({"inputs": [">>rus<< "+en_text, ">>tat<< "+en_text, ">>deu<< "+en_text,],}, API_URL_ta)
        tr_test=tuple(trans_ta())
        for tt in tr_test:
            st.write(str(tt['translation_text']))
            

st.title('Распознавание объектов с переводом на разные языки')
x_image = Image.open(io.BytesIO(load_image()))
st.image(x_image) 
result = st.button('Распознать изображение')
if result:
    with x_image  as image:
        preds = predict(image)
        st.write('**Результаты распознавания:**')
        print_predictions(preds)

