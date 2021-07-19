# import sys
# import malaya
# import logging

# logging.basicConfig(level=logging.DEBUG)

# text = 'Jabatan Penjara Malaysia diperuntukkan RM20 juta laksana program pembangunan Insan kepada banduan. Majikan yang menggaji bekas banduan, bekas penagih dadah diberi potongan cukai tambahan sehingga 2025.'


# def test_multinomial_emotion():
#     try:
#         malaya.utils.delete_cache('emotion/multinomial')
#     except BaseException:
#         pass
#     model = malaya.emotion.multinomial()
#     model.predict_proba([text])
#     malaya.utils.delete_cache('emotion/multinomial')


# def test_multinomial_sentiment():
#     try:
#         malaya.utils.delete_cache('sentiment/multinomial')
#     except BaseException:
#         pass
#     model = malaya.sentiment.multinomial()
#     model.predict_proba([text])
#     model.predict_proba([text], add_neutral=True)
#     malaya.utils.delete_cache('sentiment/multinomial')


# def test_multinomial_subjectivity():
#     try:
#         malaya.utils.delete_cache('subjective/multinomial')
#     except BaseException:
#         pass
#     model = malaya.subjectivity.multinomial()
#     model.predict_proba([text])
#     malaya.utils.delete_cache('subjective/multinomial')


# def test_multinomial_toxicity():
#     try:
#         malaya.utils.delete_cache('toxicity/multinomial')
#     except BaseException:
#         pass
#     model = malaya.toxicity.multinomial()
#     model.predict_proba([text])
#     malaya.utils.delete_cache('toxicity/multinomial')


# def test_transformer_emotion():
#     models = malaya.emotion.available_transformer()
#     for m in models.index:
#         print(m)
#         model = malaya.emotion.transformer(model=m, gpu_limit=0.3)
#         print(model.predict_proba([text]))
#         print(model.predict_words(text, visualization=False))
#         malaya.utils.delete_cache(f'emotion/{m}')
#         del model


# def test_transformer_sentiment():
#     models = malaya.sentiment.available_transformer()
#     for m in models.index:
#         print(m)
#         model = malaya.sentiment.transformer(model=m, gpu_limit=0.3)
#         print(model.predict_proba([text], add_neutral=True))
#         print(model.predict_words(text, visualization=False))
#         malaya.utils.delete_cache(f'sentiment/{m}')
#         del model


# def test_transformer_relevancy():
#     models = malaya.relevancy.available_transformer()
#     for m in models.index:
#         print(m)
#         model = malaya.relevancy.transformer(model=m, gpu_limit=0.3)
#         print(model.predict_proba([text]))
#         malaya.utils.delete_cache(f'relevancy/{m}')
#         del model


# def test_transformer_subjectivity():
#     models = malaya.subjectivity.available_transformer()
#     for m in models.index:
#         print(m)
#         model = malaya.subjectivity.transformer(model=m, gpu_limit=0.3)
#         print(model.predict_proba([text]))
#         print(model.predict_words(text, visualization=False))
#         malaya.utils.delete_cache(f'subjectivity/{m}')
#         del model


# def test_transformer_toxicity():
#     models = malaya.toxicity.available_transformer()
#     for m in models.index:
#         print(m)
#         model = malaya.toxicity.transformer(model=m, gpu_limit=0.3)
#         print(model.predict_proba([text]))
#         print(model.predict_words(text, visualization=False))
#         malaya.utils.delete_cache(f'toxicity/{m}')
#         del model
