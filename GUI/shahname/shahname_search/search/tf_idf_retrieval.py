from .get_document import preprocess_sent
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

remove_stopword = True
VERSE_DIST = 0
if remove_stopword:
    addr = "data/preprocessed_doc_stopword_removed.pkl"
else:
    addr = "data/preprocessed_doc.pkl"
with open(addr,'rb') as f:
    docs = pickle.load(f)
new_docs = {}
dataset = []
doclist = []
doctext = []
for doc in docs:
    processed_verses = [processed_verse[0] + processed_verse[1] for processed_verse, _ in docs[doc]]
    actual_verses = [actual_verse for _, actual_verse in docs[doc]]
    for i in range(len(processed_verses)):
        new_key = doc + (i,)
        processed = []
        actuals = []
        for j in range(max(0,i-VERSE_DIST),min(i+VERSE_DIST+1,len(processed_verses))):
            processed += processed_verses[j] 
            actuals.append(actual_verses[j])
        dataset.append(' '.join(processed))
        doclist.append(new_key)
        doctext.append('\n'.join(actuals))
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(dataset)
terms = vectorizer.get_feature_names_out()
terms_index = {terms[i]:i for i in range(len(terms))}



def search(query, k=10):
    # search and find related poems to query.
    # boolean method: find all terms
    # return at most k best results
    query = preprocess_sent(query, remove_stopword=remove_stopword)
    vector = [0 for i in range(len(terms))]
    for term in query:
        if term in terms_index:
            vector[terms_index[term]] += 1
    vector = np.array(vector)
    result = tfidf_matrix.dot(vector.transpose())
    inds = np.argsort(-result.transpose())[:k]
    return [doctext[i] for i in inds]