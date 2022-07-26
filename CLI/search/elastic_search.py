from get_document import preprocess_sent
import pickle
from elasticsearch import Elasticsearch


addr = "preprocessed_doc.pkl"
with open(addr, 'rb') as f:
    docs = pickle.load(f)
doclist = []
for doc in docs:
    processed_verses = [processed_verse[0] + processed_verse[1]
                        for processed_verse, _ in docs[doc]]
    actual_verses = [actual_verse for _, actual_verse in docs[doc]]
    for i in range(len(processed_verses)):
        processed = ' '.join(processed_verses[i])
        actual = actual_verses[i]
        doclist.append({
            'chapter': doc[1],
            'part': doc[2],
            'verse': i,
            'clean_text': processed,
            'actual_text': actual
        })

INDEX_NAME = "shahname"


es = Elasticsearch('http://localhost:9200')


def init_elastic():
    i = 1
    for doc in doclist:
        es.index(index=INDEX_NAME, id=i, document=doc)
        i += 1


if not es.indices.exists(index=INDEX_NAME):
    init_elastic()


def search(query, k=10):
    # search and find related poems to query.
    # boolean method: find all terms
    # return at most k best results
    query = preprocess_sent(query)
    query = ' '.join(query)
    ret = es.search(index=INDEX_NAME, query={"match": {'clean_text': query}})
    scored_docs = [(x['_score'], (x['_source']['actual_text'], x['_source']))
                   for x in ret['hits']['hits']]
    scored_docs = sorted(scored_docs, reverse=True)[:min(k, len(scored_docs))]
    return [doc[1][0] for doc in scored_docs]
