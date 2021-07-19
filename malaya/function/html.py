import string
import os
import json
import numpy as np
import shutil
from malaya.function.server import serve

try:
    from html import escape
except ImportError:
    from cgi import escape

_color_sentiment = {
    'positive': 'rgb(143, 255, 113)',
    'neutral': 'rgb(255, 238, 109)',
    'negative': 'rgb(255, 139, 118)',
}

_color_relevancy = {
    'relevant': 'rgb(143, 255, 113)',
    'not relevant': 'rgb(255, 139, 118)',
}

_color_emotion = {
    'anger': 'rgb(254, 0, 20)',
    'fear': 'rgb(168, 103, 172)',
    'happy': 'rgb(0, 211, 239)',
    'love': 'rgb(255, 70, 198)',
    'sadness': 'rgb(255, 206, 0)',
    'surprise': 'rgb(255, 244, 0)',
    'neutral': 'rgb(255, 255, 255)',
}
_color_toxic = {
    'severe toxic': 'rgb(244, 248, 0)',
    'obscene': 'rgb(248, 34, 13)',
    'threat': 'rgb(0, 238, 241)',
    'insult': 'rgb(248, 95, 34)',
    'identity attack': 'rgb(230, 183, 0)',
    'indian': 'rgb(96, 187, 173)',
    'malay': 'rgb(66, 146, 39)',
    'chinese': 'rgb(229, 169, 70)',
    'neutral': 'rgb(255, 255, 255)',
}


def _sentiment_mark(text, negative, positive, neutral, attention, label):
    return (
        "<mark style='background-color:%s' class='tooltipped' data-position='bottom' data-tooltip=\"Positive <i class='em em-smiley_cat'></i> %.3f<br>Neutral <i class='em em-cat'></i> %.3f<br>Negative <i class='em em-pouting_cat'></i> %.3f<br>Attention <i class='em em-warning'></i> %.3f\">%s</mark>" %
        (_color_sentiment[label],
         positive,
         neutral,
         negative,
         attention,
         text,
         ))


def _relevancy_mark(text, negative, positive, attention, label):
    return (
        "<mark style='background-color:%s' class='tooltipped' data-position='bottom' data-tooltip=\"Relevant <i class='em em-information_source'></i> %.3f<br>Not relevant <i class='em em-lying_face'></i> %.3f<br>Attention <i class='em em-warning'></i> %.3f\">%s</mark>" %
        (_color_relevancy[label],
         positive,
         negative,
         attention,
         text))


def _toxic_mark(
    text,
    severe_toxic,
    obscene,
    threat,
    insult,
    identity_hate,
    indian,
    malay,
    chinese,
    attention,
    label,
):
    return (
        "<mark style='background-color:%s' class='tooltipped' data-position='bottom' data-tooltip=\"Severe Toxic <i class='em em-face_vomiting'></i><i class='em em-face_vomiting'></i> %.3f<br>Obscene <i class='em em-confounded'></i> %.3f<br>Threat <i class='em em-fearful'></i> %.3f<br>Insult <i class='em em-boar'></i> %.3f<br>Identity attack <i class='em em-cry'></i> %.3f<br>Indian <i class='em em-skin-tone-6'></i> %.3f<br>Malay <i class='em em-skin-tone-4'></i> %.3f<br>Chinese <i class='em em-skin-tone-2'></i> %.3f<br>Attention <i class='em em-warning'></i> %.3f<br>\">%s</mark>" %
        (_color_toxic[label],
         severe_toxic,
         obscene,
         threat,
         insult,
         identity_hate,
         indian,
         malay,
         chinese,
         attention,
         text,
         ))


def _emotion_mark(
    text, anger, fear, happy, love, sadness, surprise, attention, label
):
    return (
        "<mark style='background-color:%s' class='tooltipped' data-position='bottom' data-tooltip=\"Anger <i class='em em-angry'></i> %.3f<br>Fear <i class='em em-fearful'></i> %.3f<br>Happy <i class='em em-smile'></i> %.3f<br>Love <i class='em em-heart_eyes'></i> %.3f<br>Sadness <i class='em em-white_frowning_face'></i> %.3f<br>Surprise <i class='em em-dizzy_face'></i> %.3f<br>Attention <i class='em em-warning'></i> %.3f<br>\">%s</mark>" %
        (_color_emotion[label],
         anger,
         fear,
         happy,
         love,
         sadness,
         surprise,
         attention,
         text,
         ))


def _render_binary(data, notebook_mode=False):
    index_negative = data['barplot']['x'].index('negative')
    index_positive = data['barplot']['x'].index('positive')
    index_neutral = data['barplot']['x'].index('neutral')
    sentiment_mark = []
    for k, v in data['word'].items():
        sentiment_mark.append(
            _sentiment_mark(
                k,
                v[index_negative],
                v[index_positive],
                v[index_neutral],
                data['alphas'][k],
                data['barplot']['x'][np.argmax(v)],
            )
        )
    sentiment_mark = ' '.join(sentiment_mark)
    this_dir = os.path.dirname(__file__)

    if notebook_mode:
        js_location, css_location = _upload_jupyter()
    else:
        js_location = 'static/echarts.min.js'
        css_location = 'static/admin-materialize.min.css'

    with open(os.path.join(this_dir, 'web', 'index.html')) as _file:
        template = string.Template(_file.read())

    template = template.substitute(
        label=escape(data['module']),
        p=sentiment_mark,
        barplot_positive=escape(
            json.dumps(int(data['barplot']['y'][index_positive]))
        ),
        barplot_neutral=escape(
            json.dumps(int(data['barplot']['y'][index_neutral]))
        ),
        barplot_negative=escape(
            json.dumps(int(data['barplot']['y'][index_negative]))
        ),
        histogram_x=escape(json.dumps(data['histogram']['x'].tolist())),
        histogram_y=escape(json.dumps(data['histogram']['y'].tolist())),
        attention_x=escape(json.dumps(data['attention']['x'].tolist())),
        attention_y=escape(json.dumps(data['attention']['y'].tolist())),
        css_location=css_location,
        js_location=js_location,
    )
    if notebook_mode:
        from IPython.display import display, HTML

        display(HTML(template))
    else:
        serve(template)


def _render_relevancy(data, notebook_mode=False):
    index_negative = data['barplot']['x'].index('not relevant')
    index_positive = data['barplot']['x'].index('relevant')
    relevancy_mark = []
    for k, v in data['word'].items():
        relevancy_mark.append(
            _relevancy_mark(
                k,
                v[index_negative],
                v[index_positive],
                data['alphas'][k],
                data['barplot']['x'][np.argmax(v)],
            )
        )
    relevancy_mark = ' '.join(relevancy_mark)
    this_dir = os.path.dirname(__file__)

    if notebook_mode:
        js_location, css_location = _upload_jupyter()
    else:
        js_location = 'static/echarts.min.js'
        css_location = 'static/admin-materialize.min.css'

    with open(os.path.join(this_dir, 'web', 'index_relevancy.html')) as _file:
        template = string.Template(_file.read())

    template = template.substitute(
        label=escape(data['module']),
        p=relevancy_mark,
        barplot_positive=escape(
            json.dumps(int(data['barplot']['y'][index_positive]))
        ),
        barplot_negative=escape(
            json.dumps(int(data['barplot']['y'][index_negative]))
        ),
        histogram_x=escape(json.dumps(data['histogram']['x'].tolist())),
        histogram_y=escape(json.dumps(data['histogram']['y'].tolist())),
        attention_x=escape(json.dumps(data['attention']['x'].tolist())),
        attention_y=escape(json.dumps(data['attention']['y'].tolist())),
        css_location=css_location,
        js_location=js_location,
    )
    if notebook_mode:
        from IPython.display import display, HTML

        display(HTML(template))
    else:
        serve(template)


def _render_toxic(data, notebook_mode=False):
    index_severe_toxic = data['barplot']['x'].index('severe toxic')
    index_obscene = data['barplot']['x'].index('obscene')
    index_threat = data['barplot']['x'].index('threat')
    index_insult = data['barplot']['x'].index('insult')
    index_identity_hate = data['barplot']['x'].index('identity attack')
    index_indian = data['barplot']['x'].index('indian')
    index_malay = data['barplot']['x'].index('malay')
    index_chinese = data['barplot']['x'].index('chinese')
    toxic_mark = []
    for k, v in data['word'].items():
        where = np.where(np.array(v) >= 0.5)[0].shape[0]
        if where:
            where = data['barplot']['x'][np.argmax(v)]
        else:
            where = 'neutral'
        toxic_mark.append(
            _toxic_mark(
                k,
                v[index_severe_toxic],
                v[index_obscene],
                v[index_threat],
                v[index_insult],
                v[index_identity_hate],
                v[index_indian],
                v[index_malay],
                v[index_chinese],
                data['alphas'][k],
                where,
            )
        )
    toxic_mark = ' '.join(toxic_mark)
    this_dir = os.path.dirname(__file__)

    if notebook_mode:
        js_location, css_location = _upload_jupyter()
    else:
        js_location = 'static/echarts.min.js'
        css_location = 'static/admin-materialize.min.css'

    with open(os.path.join(this_dir, 'web', 'index_toxic.html')) as _file:
        template = string.Template(_file.read())

    template = template.substitute(
        label=escape(data['module']),
        p=toxic_mark,
        barplot_severe_toxic=escape(
            json.dumps(int(data['barplot']['y'][index_severe_toxic]))
        ),
        barplot_obscene=escape(
            json.dumps(int(data['barplot']['y'][index_obscene]))
        ),
        barplot_threat=escape(
            json.dumps(int(data['barplot']['y'][index_threat]))
        ),
        barplot_insult=escape(
            json.dumps(int(data['barplot']['y'][index_insult]))
        ),
        barplot_identity_hate=escape(
            json.dumps(int(data['barplot']['y'][index_identity_hate]))
        ),
        barplot_indian=escape(
            json.dumps(int(data['barplot']['y'][index_indian]))
        ),
        barplot_malay=escape(
            json.dumps(int(data['barplot']['y'][index_malay]))
        ),
        barplot_chinese=escape(
            json.dumps(int(data['barplot']['y'][index_chinese]))
        ),
        histogram_x=escape(json.dumps(data['histogram']['x'].tolist())),
        histogram_y=escape(json.dumps(data['histogram']['y'].tolist())),
        attention_x=escape(json.dumps(data['attention']['x'].tolist())),
        attention_y=escape(json.dumps(data['attention']['y'].tolist())),
        css_location=css_location,
        js_location=js_location,
    )
    if notebook_mode:
        from IPython.display import display, HTML

        display(HTML(template))
    else:
        serve(template)


def _render_emotion(data, notebook_mode=False):
    index_anger = data['barplot']['x'].index('anger')
    index_fear = data['barplot']['x'].index('fear')
    index_happy = data['barplot']['x'].index('happy')
    index_love = data['barplot']['x'].index('love')
    index_sadness = data['barplot']['x'].index('sadness')
    index_surprise = data['barplot']['x'].index('surprise')
    emotion_mark = []
    for k, v in data['word'].items():
        where = np.where(np.array(v) >= 0.3)[0].shape[0]
        if where:
            where = data['barplot']['x'][np.argmax(v)]
        else:
            where = 'neutral'
        emotion_mark.append(
            _emotion_mark(
                k,
                v[index_anger],
                v[index_fear],
                v[index_happy],
                v[index_love],
                v[index_sadness],
                v[index_surprise],
                data['alphas'][k],
                where,
            )
        )
    emotion_mark = ' '.join(emotion_mark)
    this_dir = os.path.dirname(__file__)

    if notebook_mode:
        js_location, css_location = _upload_jupyter()
    else:
        js_location = 'static/echarts.min.js'
        css_location = 'static/admin-materialize.min.css'

    with open(os.path.join(this_dir, 'web', 'index_emotion.html')) as _file:
        template = string.Template(_file.read())

    template = template.substitute(
        label=escape(data['module']),
        p=emotion_mark,
        barplot_anger=escape(
            json.dumps(int(data['barplot']['y'][index_anger]))
        ),
        barplot_fear=escape(
            json.dumps(int(data['barplot']['y'][index_fear]))
        ),
        barplot_happy=escape(
            json.dumps(int(data['barplot']['y'][index_happy]))
        ),
        barplot_love=escape(
            json.dumps(int(data['barplot']['y'][index_love]))
        ),
        barplot_sadness=escape(
            json.dumps(int(data['barplot']['y'][index_sadness]))
        ),
        barplot_surprise=escape(
            json.dumps(int(data['barplot']['y'][index_surprise]))
        ),
        histogram_x=escape(json.dumps(data['histogram']['x'].tolist())),
        histogram_y=escape(json.dumps(data['histogram']['y'].tolist())),
        attention_x=escape(json.dumps(data['attention']['x'].tolist())),
        attention_y=escape(json.dumps(data['attention']['y'].tolist())),
        css_location=css_location,
        js_location=js_location,
    )
    if notebook_mode:
        from IPython.display import display, HTML

        display(HTML(template))
    else:
        serve(template)


def _attention(attn_data):
    from IPython.core.display import display, HTML, Javascript

    vis_html = """
          <span style="user-select:none">
            Layer: <select id="layer"></select>
          </span>
          <div id='vis'></div>
        """

    display(HTML(vis_html))
    this_dir = os.path.dirname(__file__)
    vis_js = open(
        os.path.join(this_dir, 'web', 'static', 'head_view.js')
    ).read()
    params = {'attention': attn_data, 'default_filter': 'all'}
    display(Javascript('window.params = %s' % json.dumps(params)))
    display(Javascript(vis_js))


def _upload_jupyter():
    location = os.getcwd()

    this_dir = os.path.dirname(__file__)

    js_location = os.path.join(this_dir, 'web', 'static', 'echarts.min.js')
    css_location = os.path.join(
        this_dir, 'web', 'static', 'admin-materialize.min.css'
    )

    shutil.copyfile(js_location, './echarts.min.js')
    shutil.copyfile(css_location, './admin-materialize.min.css')

    return 'echarts.min.js', 'admin-materialize.min.css'
