from model import Model
import pandas as pd


class Classificator:
    def __init__(self):
        self.model = Model.load_classificator()

    def cat_rule(self, row, cat_ruler):
        if cat_ruler.get(str(row['channel_id'])[-10:]):
            return cat_ruler[str(row['channel_id'])[-10:]]
        return row['category']

    def predict(self, data, cat_ruler):
        data['clear_text'] = data['clear_text'].apply(lambda x: str(x))
        prediction = self.model(data['clear_text'].values.tolist())
        prediction = pd.DataFrame(prediction)
        data['category'] = prediction['label'].tolist()
        data['category'] = data.apply(self.cat_rule, args=(cat_ruler, ), axis=1)
        return data
