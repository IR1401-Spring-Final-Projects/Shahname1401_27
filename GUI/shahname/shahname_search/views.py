from django.shortcuts import render
from prometheus_client import Enum
from .search import elastic_search as elastic
from .search import boolean_retrieval as boolean
from .search import tf_idf_retrieval as tfidf
from .search import transformer_retrieval as transformer
from .search import fasttext_retrieval as fasttext

class Method(Enum):
    ELASTIC = "elastic"
    BOOL = 'boolean'
    TFIDF = "tfidf"
    TRANS = "transformer"
    FAST = "fasttext"

def search(request, method: str = Method.BOOL):
    results = []
    query = None
    if request.method == "GET":
        query = request.GET.get('search')
        if query == '':
            query = 'None'
    srch = None
    if query != None:
        if method == Method.ELASTIC:
            srch = elastic.search
        if method == Method.BOOL:
            srch = boolean.search
        if method == Method.TFIDF:
            srch = tfidf.search
        if method == Method.FAST:
            srch = fasttext.search
        if method == Method.TRANS:
            srch = transformer.search
        if srch != None:
            results = srch(query)
    
    return render(request, 'search.html', {'query': query, 'results': results, 'method': method})
