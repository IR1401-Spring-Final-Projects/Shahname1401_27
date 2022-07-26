from transformers import AutoConfig, AutoTokenizer, AutoModel, TFAutoModel
import pickle as pkl
import torch
from .get_document import preprocess_sent

def get_sent_embedding(text):
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    token_tensors = torch.tensor([indexed_tokens])
    if torch.cuda.is_available():
        token_tensors = token_tensors.cuda()
    out = model(token_tensors)
    return out[2][-2].mean(1).squeeze()

# load model and tokenizer
# this will download the model for you if you don't have it which is around 500MB
model_name_or_path = "HooshvareLab/bert-fa-zwnj-base"
config = AutoConfig.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModel.from_pretrained(model_name_or_path, output_hidden_states=True)
model.eval()
filepath = "data/trans_embeddings.pt"
with open(filepath, 'rb') as f:
    if torch.cuda.is_available():
        embeddings = torch.load(f)
    else:
        embeddings = torch.load(f,map_location=torch.device('cpu'))
with open('data/trans_verses.pkl','rb') as f:
    original_beyts = pkl.load(f)

def search(query, k=10):
    normalized_query = ' '.join(preprocess_sent(query,True))
    query_embedding = get_sent_embedding(normalized_query)
    normalized_embeddings = embeddings  / torch.linalg.norm(embeddings, dim = 1).unsqueeze(1)
    normalized_query_embedding = query_embedding / torch.linalg.norm(query_embedding)
    ret = []
    for ind in torch.topk(torch.matmul(normalized_embeddings, normalized_query_embedding), k)[1]:
        ret.append(original_beyts[ind])
    return ret
