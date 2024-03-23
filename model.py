import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, pipeline

class SentimentAnalysisHandler(object):
    # Словарь с путями к моделям и их локальными путями
    def __init__(self):
        self.model_paths = {
            'rusentiment': {
                'model': 'blanchefort/rubert-base-cased-sentiment-rusentiment',
                'local_path': './models/rusentiment/'
            },
            'engsentiment': {
                'model': 'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis',
                'local_path': './models/engsentiment/'
            }
        }

        # Загрузка и сохранение моделей и токенизаторов для русского и английского языков
        self.tokenizer_ru, self.model_ru = self.load_and_save_model(self.model_paths['rusentiment'])
        self.tokenizer_en, self.model_en = self.load_and_save_model(self.model_paths['engsentiment'])

        # Загрузка модели для анализа тональности на русском языке
        self.rus_classifier = pipeline('text-classification', model='blanchefort/rubert-base-cased-sentiment-rusentiment')

        # Загрузка модели для анализа тональности на английском языке
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.en_model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.en_tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def load_and_save_model(self, model_info):
        # Функция для загрузки и сохранения моделей и токенизаторов
        local_path = model_info['local_path']
        model_name = model_info['model']

        if not os.path.exists(local_path):
            os.makedirs(local_path, exist_ok=True)  # Создание директории, если она не существует

        # Загрузка токенизатора
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_path)
        tokenizer.save_pretrained(local_path)

        if not os.path.isfile(os.path.join(local_path, 'pytorch_model.bin')):
            # Загрузка модели
            model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=local_path)

            # Сохранение состояния модели в формате .bin (если надо будет деплоить на torchserve)
            torch.save(model.state_dict(), os.path.join(local_path, 'pytorch_model.bin'))

            # Сохраняется конфиг модели для последующей загрузки
            model.config.save_pretrained(local_path)
        else:
            config = AutoConfig.from_pretrained(local_path)
            model = AutoModelForSequenceClassification.from_config(config)
            model.load_state_dict(torch.load(os.path.join(local_path, 'pytorch_model.bin')))

        return tokenizer, model

    def preprocess(self, data):
        # Получение текста и языка из входных данных
        text = data.get("text")
        language = data.get("language")

        # Выбор модели и токенизатора на основе языка
        if language == "rusentiment":
            tokenizer = self.tokenizer_ru
        elif language == "engsentiment":
            tokenizer = self.tokenizer_en

        # Токенизация текста и преобразование в тензор PyTorch
        encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        return encoding, language

    def predict_sentiment_russian(self, text):
        # Функция для предсказания сентимента для русского текста
        out = self.rus_classifier(text)
        return out[0]['label'].upper()

    def predict_sentiment_english(self, text):
        # Функция для предсказания сентимента для английского текста
        inputs = self.en_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.en_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiments = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
        result = sentiments[predictions.argmax().item()]  # Получение индекса наиболее вероятной тональности и преобразование в строку
        return result

    def inference(self, data):
        # Предобработка входных данных
        encoding, language = self.preprocess(data)

        if language == "rusentiment":
            text = data.get("text")
            sentiment = self.predict_sentiment_russian(text)
            return sentiment
        elif language == "engsentiment":
            text = data.get("text")
            sentiment = self.predict_sentiment_english(text)
            return sentiment
