from fastapi import FastAPI,UploadFile
from fastapi.responses import StreamingResponse
import pandas as pd
import uvicorn
from deduplicator import Deduplicator
from preprocessing import DataPreprocessing
from classificator import Classificator
import json

app = FastAPI()
data_preprocessing = DataPreprocessing()
deduplicator = Deduplicator()
classificator = Classificator()

with open('ruler.json', 'r') as f:
    ruler = json.load(f)


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    df = pd.read_csv(file.file)
    df = df[['text', 'channel_id']]
    print('Файл успешно загружен.')
    df = df.dropna(subset='text')
    print('Началась предобработка.')
    df = data_preprocessing.data_prepare(df)
    print('Файл успешно обработан.')
    df = df.drop_duplicates(subset='clear_text')
    list_to_drop = deduplicator.drop_near_duplicates(df['clear_text'])
    print('Получен список дубликатов.')
    df = df.drop(list_to_drop, axis=0)
    print('Началась классификация.')
    df = classificator.predict(df, ruler)
    print('Классификация завершена!')
    df = df[['text', 'channel_id', 'category']]
    response = StreamingResponse(iter([df.to_csv(index=False)]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=export.csv"
    return response
