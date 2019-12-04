import malaya
from fastapi import FastAPI

app = FastAPI()

model = malaya.sentiment.transformer(
    model = 'albert', size = 'base', validate = False
)


@app.get('/')
def index(string: str = None):
    r = model.predict(string, get_proba = True)
    return {k: float(v) for k, v in r.items()}
