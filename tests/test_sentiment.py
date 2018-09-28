import malaya

def test_pretrained_bayes_sentiment():
    positive_text = 'Kerajaan negeri Kelantan mempersoalkan motif kenyataan Menteri Kewangan Lim Guan Eng yang hanya menyebut Kelantan penerima terbesar bantuan kewangan dari Kerajaan Persekutuan. Sedangkan menurut Timbalan Menteri Besarnya, Datuk Mohd Amar Nik Abdullah, negeri lain yang lebih maju dari Kelantan turut mendapat pembiayaan dan pinjaman.'
    news_sentiment = malaya.pretrained_bayes_sentiment()
    assert len(news_sentiment.predict(positive_text)) > 0

def test_pretrained_xgb_sentiment():
    positive_text = 'Kerajaan negeri Kelantan mempersoalkan motif kenyataan Menteri Kewangan Lim Guan Eng yang hanya menyebut Kelantan penerima terbesar bantuan kewangan dari Kerajaan Persekutuan. Sedangkan menurut Timbalan Menteri Besarnya, Datuk Mohd Amar Nik Abdullah, negeri lain yang lebih maju dari Kelantan turut mendapat pembiayaan dan pinjaman.'
    news_sentiment = malaya.pretrained_xgb_sentiment()
    assert len(news_sentiment.predict(positive_text)) > 0

def test_pretrained_bayes_sentiment_batch():
    positive_text = 'Kerajaan negeri Kelantan mempersoalkan motif kenyataan Menteri Kewangan Lim Guan Eng yang hanya menyebut Kelantan penerima terbesar bantuan kewangan dari Kerajaan Persekutuan. Sedangkan menurut Timbalan Menteri Besarnya, Datuk Mohd Amar Nik Abdullah, negeri lain yang lebih maju dari Kelantan turut mendapat pembiayaan dan pinjaman.'
    news_sentiment = malaya.pretrained_bayes_sentiment()
    assert len(news_sentiment.predict_batch([positive_text,positive_text])) > 0

def test_bahdanau_sentiment():
    malaya.get_available_sentiment_models()
    negative_text = 'kerajaan sebenarnya sangat bencikan rakyatnya, minyak naik dan segalanya'
    news_sentiment = malaya.deep_sentiment('bahdanau')
    assert len(news_sentiment.predict(negative_text)['attention']) > 1

def test_attention_sentiment():
    malaya.get_available_sentiment_models()
    negative_text = 'kerajaan sebenarnya sangat bencikan rakyatnya, minyak naik dan segalanya'
    news_sentiment = malaya.deep_sentiment('hierarchical')
    assert len(news_sentiment.predict(negative_text)['attention']) > 1

def test_luong_sentiment():
    malaya.get_available_sentiment_models()
    negative_text = 'kerajaan sebenarnya sangat bencikan rakyatnya, minyak naik dan segalanya'
    news_sentiment = malaya.deep_sentiment('luong')
    assert len(news_sentiment.predict(negative_text)['attention']) > 1

def test_normal_sentiment():
    malaya.get_available_sentiment_models()
    negative_text = 'kerajaan sebenarnya sangat bencikan rakyatnya, minyak naik dan segalanya'
    news_sentiment = malaya.deep_sentiment('bidirectional')
    assert news_sentiment.predict(negative_text)['negative'] > 0

def test_fasttext_sentiment():
    malaya.get_available_sentiment_models()
    negative_text = 'kerajaan sebenarnya sangat bencikan rakyatnya, minyak naik dan segalanya'
    news_sentiment = malaya.deep_sentiment('fast-text')
    assert news_sentiment.predict(negative_text)['negative'] > 0

def test_normal_sentiment_batch():
    malaya.get_available_sentiment_models()
    negative_text = 'kerajaan sebenarnya sangat bencikan rakyatnya, minyak naik dan segalanya'
    news_sentiment = malaya.deep_sentiment('bidirectional')
    assert len(news_sentiment.predict_batch([negative_text,negative_text])) > 0

def test_fasttext_sentiment_batch():
    malaya.get_available_sentiment_models()
    negative_text = 'kerajaan sebenarnya sangat bencikan rakyatnya, minyak naik dan segalanya'
    news_sentiment = malaya.deep_sentiment('fast-text')
    assert len(news_sentiment.predict_batch([negative_text,negative_text])) > 0

def test_bayes_sentiment():
    import pandas as pd
    df = pd.read_csv('tests/02032018.csv',sep=';')
    df = df.iloc[3:,1:]
    df.columns = ['text','label']
    dataset = [[df.iloc[i,0],df.iloc[i,1]] for i in range(df.shape[0])]
    bayes=malaya.bayes_sentiment(dataset)
    assert len(bayes.predict(dataset[0][0])) > 0

def test_bayes_sentiment_bow_nosplit():
    import pandas as pd
    df = pd.read_csv('tests/02032018.csv',sep=';')
    df = df.iloc[3:,1:]
    df.columns = ['text','label']
    dataset = [[df.iloc[i,0],df.iloc[i,1]] for i in range(df.shape[0])]
    bayes=malaya.bayes_sentiment(dataset, vector = 'bow', split_size = None)
    assert len(bayes.predict(dataset[0][0])) > 0

def test_bayes_sentiment_location():
    bayes = malaya.bayes_sentiment('tests/local')
    assert len(bayes.predict('saya suka kerajaan dan anwar ibrahim')) > 0
