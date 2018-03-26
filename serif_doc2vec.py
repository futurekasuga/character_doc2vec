#coding: UTF-8

from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import MeCab
import sys

#MeCabの固有表現辞書を使う
mecab = MeCab.Tagger("-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")
trainings = []
#キャラとセリフが空白区切りのデータを使う
chara_serif_corpus = sys.argv[1]
with open(chara_serif_corpus,'r') as f:
    print('parse start')
    for i,line in enumerate(f):
        line = line.rstrip().split(' ')
        name = line[0]
        serif = line[1]
        mecab_result = mecab.parse(serif)
        words = mecab_result.split(' ')
        training = TaggedDocument(words = words,tags = [name + str(i)])
        trainings.append(training)
print('parse end')
print('learn start')
model = Doc2Vec(documents= trainings, size=300, window=15, min_count=1,workers=4, iter=400, dbow_words=1, negative=5)
model.save("{0}doc2vec.model".format(chara_serif_corpus))
print('end')
