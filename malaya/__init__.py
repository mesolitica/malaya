from .normalizer import naive_normalizer, basic_normalizer
from .num2word import to_cardinal,to_ordinal,to_ordinal_num,to_currency,to_year
from .pos_entities import deep_pos_entities, naive_pos
from .sentiment import deep_sentiment, bayes_sentiment, pretrained_bayes_sentiment
from .stemmer import naive_stemmer
from .topic_modelling import lda_topic_modelling, nmf_topic_modelling, lsa_topic_modelling
from .word2vec import malaya_word2vec, Word2Vec

from pathlib import Path
home = str(Path.home())+'/Malaya'

try:
    if not os.path.exists(home):
        os.makedirs(home)
except:
    print('cannot make directory for cache, exiting.')
    sys.exit(1)
