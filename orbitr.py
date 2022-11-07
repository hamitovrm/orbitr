import io
import torch
import requests
import streamlit as st
from PIL import Image
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel

loc = "ydshieh/vit-gpt2-coco-en"

feature_extractor = ViTFeatureExtractor.from_pretrained(loc)
tokenizer = AutoTokenizer.from_pretrained(loc)
model = VisionEncoderDecoderModel.from_pretrained(loc)
model.eval()


def predict(image):
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    with torch.no_grad():
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True).sequences
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


API_URL_ta = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-mul"
headers = {"Authorization": f"Bearer {'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'}"}

def translate(payload, API_URL):
	response = requests.post(API_URL, headers=headers, json=payload )
	return response.json
	
def preprocess_image(img):
    img = img.resize((224, 224))
    #x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)
    return img#x

def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

def print_predictions(preds):
    for cl in preds:
        #st.write(str(cl).replace('_'," "))
        en_text=str(cl).replace('_'," ")
        trans_ta = translate({"inputs": [">>rus<< "+en_text, ">>tat<< "+en_text, ">>deu<< "+en_text,],}, API_URL_ta)
        tr_test=tuple(trans_ta())
        for tt in tr_test:
            #st.write(str(tt['translation_text']))
            st.write(str(tt))
            
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
with Image.open(requests.get(url, stream=True).raw) as image:
    preds = predict(image)
    st.write(preds)
    print_predictions(preds)             
            
st.title('Распознавание объектов с переводом на разные языки')
img = load_image()
result = st.button('Распознать изображение')

if result:
   x=preprocess_image(img):
   preds = predict(x)
   st.write('**Результаты распознавания:**')
   print_predictions(preds)

