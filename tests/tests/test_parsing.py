import sys
import malaya
import logging

logging.basicConfig(level=logging.DEBUG)

string = 'Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar sekiranya mengantuk ketika memandu.'


def test_constituency():
    models = malaya.constituency.available_transformer()
    for m in models.index:
        print(m)
        model = malaya.constituency.transformer(model=m, gpu_limit=0.3)
        print(model.parse_tree(string))
        malaya.utils.delete_cache(f'constituency/{m}')
        del model


def test_dependency_v2():
    models = malaya.dependency.available_transformer()
    for m in models.index:
        print(m)
        model = malaya.dependency.transformer(model=m, gpu_limit=0.3)
        d_object, tagging, indexing = model.predict(string)
        malaya.utils.delete_cache(f'dependency-v2/{m}')
        del model


def test_dependency_v1():
    models = malaya.dependency.available_transformer(version='v1')
    for m in models.index:
        print(m)
        model = malaya.dependency.transformer(version='v1', model=m, gpu_limit=0.3)
        d_object, tagging, indexing = model.predict(string)
        malaya.utils.delete_cache(f'dependency/{m}')
        del model
