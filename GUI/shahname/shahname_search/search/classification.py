
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
    out = pmodel(token_tensors)
    return out[2][-2].mean(1).squeeze()



# load model and tokenizer
# this will download the model for you if you don't have it which is around 500MB
model_name_or_path = "HooshvareLab/bert-fa-zwnj-base"
config = AutoConfig.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
pmodel = AutoModel.from_pretrained(model_name_or_path, output_hidden_states=True)
pmodel.eval()
filepath = "data/trans_embeddings.pt"
with open(filepath, 'rb') as f:
    if torch.cuda.is_available():
        embeddings = torch.load(f)
    else:
        embeddings = torch.load(f,map_location=torch.device('cpu'))
with open('data/trans_verses.pkl','rb') as f:
    original_beyts = pkl.load(f)

with open('data/classmodel.pt', 'rb') as f:
    if torch.cuda.is_available():
        model = torch.load(f)
    else:
        model = torch.load(f,map_location=torch.device('cpu'))

with open('data/rclasses.pkl','rb') as f:
    reverse_classes = pkl.load(f)

def respond_to_query(text):
  processed_q = ' '.join(preprocess_sent(text))
  q_embedding = get_sent_embedding(processed_q).unsqueeze(0)
  return reverse_classes[(model(q_embedding).squeeze().argmax().item())]

