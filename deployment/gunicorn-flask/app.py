from flask import Flask, request, jsonify
import malaya

app = Flask(__name__)

model = malaya.sentiment.transformer(
    model = 'albert', size = 'base', validate = False
)


@app.route('/', methods = ['GET'])
def index():
    string = request.args.get('string')
    r = model.predict(string, get_proba = True)
    return {k: float(v) for k, v in r.items()}


application = app
