import io
import requests
import streamlit as st
from PIL import Image
import numpy as np
import jax
from transformers import ViTFeatureExtractor, AutoTokenizer, FlaxVisionEncoderDecoderModel


API_URL_ta = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-mul"
headers = {"Authorization": f"Bearer {'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'}"}

def translate(payload, API_URL):
	response = requests.post(API_URL, headers=headers, json=payload )
	return response.json
	
loc = "ydshieh/vit-gpt2-coco-en"

feature_extractor = ViTFeatureExtractor.from_pretrained(loc)
tokenizer = AutoTokenizer.from_pretrained(loc)
model = FlaxVisionEncoderDecoderModel.from_pretrained(loc)

gen_kwargs = {"max_length": 16, "num_beams": 4}


# This takes sometime when compiling the first time, but the subsequent inference will be much faster

def generate(pixel_values):
    output_ids = model.generate(pixel_values, **gen_kwargs).sequences
    return output_ids
    
    
def predict(image):
    pixel_values = feature_extractor(images=image, return_tensors="np").pixel_values
    output_ids = generate(pixel_values)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds] 
    return preds


def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

def print_predictions(preds):
    classes = decode_predictions(preds, top=3)[0]
    for cl in classes:
        st.write(str(cl[1]).replace('_'," "), cl[2])
        en_text=str(cl[1]).replace('_'," ")
        trans_ta = translate({"inputs": [">>rus<< "+en_text, ">>tat<< "+en_text, ">>deu<< "+en_text,],}, API_URL_ta)
        tr_test=tuple(trans_ta())
        for tt in tr_test:
            st.write(str(tt['translation_text']))

st.title('Распознавание объектов с переводом на разные языки')
img = load_image()
result = st.button('Распознать изображение')
if result:
    preds = predict(img)
    st.write('**Результаты распознавания:**')
    print_predictions(preds)

