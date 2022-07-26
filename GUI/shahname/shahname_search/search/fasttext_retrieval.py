import fasttext.util
import pickle as pkl
import torch
from get_document import preprocess_sent

fasttext.util.download_model('fa', if_exists='ignore') #comment line if alraedy downloaded
ft = fasttext.load_model('cc.fa.300.bin')

filepath = "data/fasttext_embeddings_with_stopwords.pkl"
# vectorizer = TfidfVectorizer(strip_accents='unicode')
with open(filepath, 'rb') as f:
    loaded_pickle = pkl.load(f)
    embeddings, original_beyts = loaded_pickle['embeddings'], loaded_pickle['beyts']


def search(query, k=10):
    normalized_query = preprocess_sent(query, True)
    query_embedding = torch.from_numpy(ft.get_sentence_vector(''.join(normalized_query))).cuda()
    normalized_embeddings = embeddings  / torch.linalg.norm(embeddings, dim = 1).unsqueeze(1)
    normalized_query_embedding = query_embedding / torch.linalg.norm(query_embedding)
    ret = []
    for ind in torch.topk(torch.matmul(normalized_embeddings, normalized_query_embedding), k)[1]:
        ret.append(original_beyts[ind])
    return ret