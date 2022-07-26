from string import punctuation
from bs4 import BeautifulSoup
from collections import defaultdict
from hazm import *
import codecs
normalizer = Normalizer()


def get_documents():
    with open('data/shahname.htm', 'r', encoding='utf-8') as f:
        contents = f.read()
        soup = BeautifulSoup(contents, 'lxml')

    els = soup.find_all(['h1', 'h2', 'h3', 'p'])

    docs = defaultdict(list)
    currh1 = None
    currh2 = None
    currh3 = None

    for el in els:
        if 'content_h1' in el['class']:
            currh1 = el.text
            currh2 = None
            currh3 = None
        elif 'content_h2' in el['class']:
            currh2 = el.text
            currh3 = None
        elif 'content_h3' in el['class']:
            currh3 = el.text
        elif 'content_paragraph' in el['class'] and '****' in el.text:
            docs[(currh1, currh2, currh3)].append(el.text)
    return docs


old_expressions = [('ز', 'از'), ('چو', 'چون'), ('همی', 'می'), ('اندر', 'در'), ('بدو', 'به او'), ('مرا', 'من را'), ('گر', 'اگر'), ('بیامد', 'آمد'), ('بدین', 'به این'), ('ازان', 'از آن'), ('اوی', 'او'), ('کنون', 'اکنون'), ('زان', 'از آن'), ('اندرون', 'درون'), ('دگر', 'دیگر'), ('برین', 'بر این'), ('ترا', 'تو را'), ('کای', 'که ای'), ('بفرمود', 'فرمود'), ('ازین', 'از این'), ('وز', 'و از'), ('کز', 'که از'), ('پیل', 'فیل'), ('وزان', 'و از آن'), ('ورا', 'او را'), ('ازو', 'از او'), ('برفتند', 'رفتند'), ('زو', 'از او'), ('وگر', 'و اگر'), ('مر', 'مگر'), ('بدید', 'دید'), ('بگفت', 'گفت'), ('بباید', 'باید'), ('بیاورد', 'آورد'), ('بپرسید', 'پرسید'), ('بخواند', 'خواند'), ('مکن', 'نکن'), ('بزد', 'زد'), ('کزین', 'که از این'), ('براند', 'راند'), ('آنچ', 'آنجه'), ('بنهاد', 'نهاد'), ('چهر', 'چهره')]
old_expressions_map = {}
for key,val in old_expressions:
    old_expressions_map[key] = val
stopwords = [normalizer.normalize(x.strip()) for x in codecs.open('data/stopwords.txt','r','utf-8').readlines()]
punctuation = '.؟?.,،!!'


def preprocess_verse(verse, remove_stopword = False):
    parts = verse.split('****')
    return [preprocess_sent(part, remove_stopword) for part in parts]

def preprocess_sent(sent, remove_stopword = False):
    # turn old expressions to new
    sent = alter_old_expressions(sent)
    # normalize
    sent = normalizer.normalize(sent)
    # tokenize
    sent = [' '.join(word_tokenize(word)) for word in sent.split()]
    # remove punctuation
    sent = remove_punctuation(sent)
    # remove stopwords ?
    if remove_stopword:
        sent = remove_stopwords(sent)
    return sent


def remove_punctuation(wlist):
    return [word for word in wlist if word not in punctuation]


def alter_old_expressions(sent):
    sent = sent.split()
    new_sent = []
    for part in sent:
        if part in old_expressions_map:
            new_sent.append(old_expressions_map[part])
        else:
            new_sent.append(part)
    return ' '.join(new_sent)

def remove_stopwords(sent):
    return [word for word in sent if word not in stopwords]

def preprocess_docs(docs, remove_stopword = False):
    new_docs = {}
    for doc in docs:
        sents = docs[doc]
        new_sents = []
        for sent in sents:
            new_sents.append((preprocess_verse(sent,remove_stopword), sent))
        new_docs[doc] = new_sents
    return new_docs

def get_preprocessed_docs(remove_stopword = False):
    return preprocess_docs(get_documents(),remove_stopword)


import pickle

def save_docs():
    with open('preprocessed_doc.pkl','wb') as f:
        pickle.dump(get_preprocessed_docs(),f)
    with open('preprocessed_doc_stopword_removed.pkl','wb') as f:
        pickle.dump(get_preprocessed_docs(remove_stopword=True),f)
    