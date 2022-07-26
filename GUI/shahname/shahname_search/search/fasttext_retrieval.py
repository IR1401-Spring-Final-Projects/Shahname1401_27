import fasttext.util
import pickle as pkl
import torch
from .get_document import preprocess_sent

fasttext.util.download_model('fa', if_exists='ignore') #comment line if alraedy downloaded
ft = fasttext.load_model('cc.fa.300.bin')

filepath = "data/fast_embeddings.pt"
with open(filepath, 'rb') as f:
    if torch.cuda.is_available():
        embeddings = torch.load(f)
    else:
        embeddings = torch.load(f,map_location=torch.device('cpu'))
with open('data/fast_verses.pkl','rb') as f:
    original_beyts = pkl.load(f)

def search(query, k=10):
    normalized_query = preprocess_sent(query, True)
    query_embedding = torch.from_numpy(ft.get_sentence_vector(' '.join(normalized_query)))
    if torch.cuda.is_available():
        query_embedding = query_embedding.cuda()
    normalized_embeddings = embeddings  / torch.linalg.norm(embeddings, dim = 1).unsqueeze(1)
    normalized_query_embedding = query_embedding / torch.linalg.norm(query_embedding)
    ret = []
    for ind in torch.topk(torch.matmul(normalized_embeddings, normalized_query_embedding), k)[1]:
        ret.append(original_beyts[ind])
    return ret