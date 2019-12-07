from flask import Flask, request, jsonify
import malaya

app = Flask(__name__)

global model


@app.before_first_request
def load_model():
    global model
    model = malaya.sentiment.transformer(
        model = 'albert', size = 'base', validate = False
    )


@app.route('/', methods = ['GET'])
def index():
    strings = [request.args.get('string')] * 50
    r = model.predict_batch(strings, get_proba = True)
    return jsonify('done')


@app.route('/test', methods = ['GET'])
def test():
    return jsonify('test')


application = app
