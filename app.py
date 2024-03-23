from flask import Flask, request, jsonify, render_template
from model import SentimentAnalysisHandler
from langdetect import detect

# Создание Flask экземпляра
app = Flask(__name__)

# Создание экземпляра SentimentAnalysisHandler из model.py
handler = SentimentAnalysisHandler()

# Декоратор для обработки GET-запросов на корневой URL
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Декоратор для обработки POST-запросов на URL /sentiment
@app.route('/sentiment', methods=['POST'])
def get_sentiment():
    # Получение JSON-данных из запроса
    data = request.get_json()
    # Извлечение текста из JSON-данных
    text = data.get('text')
    # Возвращение ошибки, если текст не предоставлен
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        # Определение языка текста
        language = 'rusentiment' if detect(text) == 'ru' else 'engsentiment'
        # Вызов метода inference для анализа тональности
        sentiment = handler.inference({"text": text, "language": language})
        # Возвращение результата анализа тональности
        return jsonify({'sentiment': sentiment})
    # Возвращение ошибки, если произошло исключение
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Запуск Flask-приложения на порту 5000 и доступного для всех интерфейсов
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
