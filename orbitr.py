import io
import torch
import requests
import streamlit as st
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer

@st.cache(allow_output_mutation=True)
def load_model():
    return VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
@st.cache(allow_output_mutation=True)
def load_feature_extractor():
    return ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
@st.cache(allow_output_mutation=True)
def load_tokenizer():
    return AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def predict_step(image):
  images = []
  if image.mode != "RGB":
     image = image.convert(mode="RGB")
  images.append(image)
  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)
  output_ids = model.generate(pixel_values, **gen_kwargs)
  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds

def translate(payload, API_URL):
	response = requests.post(API_URL, headers=headers, json=payload )
	return response.json

def load_image():
    uploaded_file = st.file_uploader(label='Загрузите изображение:')
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
        trans_ta = translate({"inputs": [">>rus<< "+en_text, ">>tat<< "+en_text, ">>deu<< "+en_text, ">>fra<< "+en_text, ">>ita<< "+en_text,]}, API_URL_ta)
        sleep_duration = 0.5
        tr_test=tuple(trans_ta())
        sleep_duration = 0.5
        st.write('рус: ', tr_test[0]["translation_text"])
        st.write('тат: ', tr_test[1]["translation_text"])
        st.write('deu: ', tr_test[2]["translation_text"])
        st.write('fra:', tr_test[3]["translation_text"])
        st.write('ita:', tr_test[4]["translation_text"])
        #for tt in tr_test:
        #    st.write(str(tt['translation_text']))
	    

st.title('Распознавание объектов с переводом на разные языки')
model = load_model()	
feature_extractor = load_feature_extractor()
tokenizer=load_tokenizer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

API_URL_ta = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-mul"
headers = {"Authorization": f"Bearer {'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'}"}

#Тестовое изображение
#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#with Image.open(requests.get(url, stream=True).raw) as image1:
#    st.image(image1)
#    preds = predict_step(image1)
#    st.write(preds)
#    print_predictions(preds)        
            
im=load_image()
result = st.button('Распознать:')
if result:
   preds = predict_step(im)
   st.write('**На картинке:**')
   st.write(str(preds))
   print_predictions(preds) 

