from get_document import preprocess_sent
from collections import defaultdict
import pickle
VERSE_DIST = 0
remove_stopword = False
if remove_stopword:
    addr = "preprocessed_doc_stopword_removed.pkl"
else:
    addr = "preprocessed_doc.pkl"
with open(addr,'rb') as f:
    docs = pickle.load(f)
new_docs = {}
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
        new_docs[new_key] = (processed,'\n'.join(actuals))

inv_index = defaultdict(set)
doc_term_freq = {}

for doc in new_docs:
    doc_term_freq[doc] = defaultdict(lambda: 0)
    processed_verse, actual_verse = new_docs[doc]
    for term in processed_verse:
        inv_index[term].add(doc)
        doc_term_freq[doc][term] += 1
    doc_max_freq = max(doc_term_freq[doc].values())
    for term in processed_verse:
        doc_term_freq[doc][term] /= doc_max_freq


def search(query, k=10):
    # search and find related poems to query.
    # boolean method: find all terms
    # return at most k best results
    query = preprocess_sent(query, remove_stopword=remove_stopword)
    ret = None
    for term in query:
        if ret is None:
            ret = inv_index[term]
        else:
            ret = ret.intersection(inv_index[term])
    scored_docs = []
    for doc in ret:
        scored_docs.append((get_doc_score(doc, query), doc))
    scored_docs = sorted(scored_docs, reverse=True)[:min(k, len(scored_docs))]
    return [(doc, new_docs[doc][1]) for _, doc in scored_docs]

def get_doc_score(doc, query):
    ret = 0
    for term in query:
        ret += doc_term_freq[doc][term]
    return ret