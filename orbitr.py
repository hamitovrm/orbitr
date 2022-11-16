import io
import torch
import requests
import streamlit as st
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

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


API_URL_ta = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-mul"
headers = {"Authorization": f"Bearer {'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'}"}

def translate(payload, API_URL):
	response = requests.post(API_URL, headers=headers, json=payload )
	return response.json

def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания',type=['jpg','png','jpeg'])
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return image_data #Image.open(image_data)
    else:
        return None

def preprocess_image(img):
    img = img.resize((224, 224))


def print_predictions(preds):
    for cl in preds:
        #st.write(str(cl).replace('_'," "))
        en_text=str(cl).replace('_'," ")
        trans_ta = translate({"inputs": [">>rus<< "+en_text, ">>tat<< "+en_text, ">>deu<< "+en_text,],}, API_URL_ta)
        tr_test=tuple(trans_ta())
        for tt in tr_test:
            st.write(str(tt['translation_text']))
            
#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#with Image.open(requests.get(url, stream=True).raw) as image1:
#    st.image(image1)
 #   preds = predict_step(image1)
  #  st.write(preds)
 #   print_predictions(preds)        
 #   st.write(type(image1))    
            
st.title('Распознавание объектов с переводом на разные языки')
im=load_image()


result = st.button('Распознать изображение')
if result:
   #im.load()
   x=preprocess_image(im)
   st.write(type(x))
   #preds = predict_step(x)
#        st.write('**Результаты распознавания:**')
#        st.write(str(preds))

