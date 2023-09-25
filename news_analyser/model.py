from sentence_transformers import SentenceTransformer
from transformers import pipeline


class Model:
    def __init__(self) -> None:
        pass

    def load_deduplicator():
        model = SentenceTransformer('./models/intfloat/multilingual-e5-small/intfloat_multilingual-e5-small')
        return model

    def load_classificator():
        model = pipeline("text-classification", model="./models/Roverandom95/my_rubert_model")
        return model
