Orbitr - Web-приложение искусственного интеллекта для ВКР онлайн-курса DevOps
Web-приложение для распознавания объектов изображений. Используются библиотеки:

TensorFlow.

Для распознавания изображений используется нейронная сеть:
https://huggingface.co/nlpconnect/vit-gpt2-image-captioning

Для перевода текста используется нейронная сеть:
https://huggingface.co/Helsinki-NLP/opus-mt-en-mul

Интерфейс реализован на базе Streamlit.

https://orbitr.streamlit.app/

Для локального запуска ввести в коммандной строке:
streamlit run orbitr.py

