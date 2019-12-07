import malaya
import json
from fastapi import FastAPI

app = FastAPI()

model = malaya.sentiment.transformer(
    model = 'albert', size = 'base', validate = False
)


@app.get('/')
def index(string: str = None):
    strings = [string] * 50
    r = model.predict_batch(strings, get_proba = True)
    return json.dumps('done')
