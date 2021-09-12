import gensim
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import pandas as pd

warnings.filterwarnings(action='ignore')

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
model.similarity('woman', 'man')

file_name = "empty_shopee_philippines.csv"
user_id = pd.read_csv(file_name, encoding="ISO-8859-1", usecols=range(0, 1))

categories = ["women clothing", "health beauty", "baby toys", "home living", "women bag", ]