import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class DataPreprocessing:

    def __init__(self) -> None:
        pass

    def clean_text(self, input_text):
        input_text = str(input_text)

        # Проверка на рекламу по ЕРИР
        ad = r'Реклама.*(ERID|Erid|erid|ОГРН)'

        if bool(re.search(ad, input_text)):
            return "РЕКЛАМА"

        # Проверка на рекламу по хэштегам
        # Список хэштегов для проверки
        hashtags_to_check = ['#нативнаяинтеграция', '#интеграция', '#реклама', '#спонсор']

        # Регулярное выражение для поиска хэштегов в тексте
        pattern = '|'.join(re.escape(tag) for tag in hashtags_to_check)

        # Проверяем, содержит ли текст один из хэштегов
        contains_hashtags = bool(re.search(pattern, input_text))

        if contains_hashtags:
            return "РЕКЛАМА"

        # URL и ссылки: далее - удаляем из текста все URL и ссылки
        clean_text = re.sub(r'http\S+', '', input_text)

        # Приводим все входные данные к нижнему регистру
        clean_text = clean_text.lower()

        clean_text = clean_text.replace('ё', 'е')
        clean_text = clean_text.replace('Ё', 'Е')

        # Заменяем перенос строки на точку
        clean_text = re.sub('\n', '. ', clean_text)

        # Убираем все лишние пробелы и тчки
        clean_text = re.sub('(\. )+', '. ', clean_text)

        # Убираем специальные символы
        clean_text = re.sub('[^а-яА-Яa-zA-Z]', ' ', clean_text)
        clean_text = re.sub('\s+', ' ', clean_text)

        # Стоп-слова: удаление стоп-слов - это стандартная практика очистки текстов
        stop_words = set(stopwords.words('russian'))
        tokens = word_tokenize(clean_text)
        tokens = [token for token in tokens if (token not in stop_words) and (len(token) > 1)]
        clean_text = ' '.join(tokens)

        return clean_text

    def data_prepare(self, data):
        data['clear_text'] = data['text'].apply(self.clean_text)
        data = data.query('clear_text != "РЕКЛАМА"')
        return data
