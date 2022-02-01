from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm.auto import tqdm
import re

a = np.array(['hello world'])
b = np.array(['hello wrld hllo'])
dic = ['hello', 'this', 'is', 'count', 'vectorizer']
dic2 = ['hello, this is count vectorizer!']

vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 4)).fit(dic)
#  обучить на векторе

print(vectorizer.get_feature_names_out())
a1 = vectorizer.transform(a)
b1 = vectorizer.transform(b)
print(cosine_similarity(a1, b1))
