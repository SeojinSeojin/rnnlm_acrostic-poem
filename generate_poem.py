# coding: utf-8

from load_dataset import load_data
from RnnlmGen import RnnlmGen
import sys
import random
sys.path.append('..')

alphaList = [chr(c) for c in range(ord('a'), ord('z')+1)]
# 출처: https://devyongsik.tistory.com/411 [DEV용식]
corpus, word_to_id, id_to_word = load_data('train')
words = word_to_id.keys()


def getWordStartWith(char):
    wordsStartWithChar = list(filter(lambda str: str[0] == char, words))
    return wordsStartWithChar[random.randint(0, len(wordsStartWithChar)-1)]


start_words = dict(map(lambda char: (char, getWordStartWith(char)), alphaList))


def create_poem(word):

    model = RnnlmGen()
    model.load_params('./Rnnlm.pkl')

    start_word = start_words[word[0]]
    start_id = word_to_id[start_word]

    skip_words = ['N', '<unk>', '$']
    skip_ids = [word_to_id[w] for w in skip_words]

    word_ids = model.generate(start_id, word, id_to_word, skip_ids)
    txt = ' '.join([id_to_word[i] for i in word_ids])
    txt = txt.replace(' <eos>', '.\n')
    return txt
