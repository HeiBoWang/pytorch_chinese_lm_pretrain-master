# from transformers import BertTokenizer, BertModel
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#
## First we initialize our model and tokenizer:
# tokenizer = BertTokenizer.from_pretrained('./result')
# model = BertModel.from_pretrained('./result')


def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


"""
param: sentence list
return: embeddings
"""


def encode(sentences, tokenizer, model):
    tokens = {'input_ids': [], 'attention_mask': []}
    data_num = len(sentences)

    for sentence in sentences:
        # 编码每个句子并添加到字典
        new_tokens = tokenizer.encode_plus(str(sentence), max_length=128,
                                           truncation=True, padding='max_length',
                                           return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])

    # 将张量列表重新格式化为一个张量
    tokens['input_ids'] = torch.stack(tokens['input_ids']).to(device)
    tokens['attention_mask'] = torch.stack(tokens['attention_mask']).to(device)
    model.eval()

    # We process these tokens through our model:
    with torch.no_grad():  # 添加这行代码
        outputs = model(**tokens)

    # odict_keys(['last_hidden_state', 'pooler_output'])

    # The dense vector representations of our text are contained within the outputs 'last_hidden_state' tensor, which we access like so:

    embeddings = outputs[0]

    # To perform this operation, we first resize our attention_mask tensor:

    attention_mask = tokens['attention_mask']
    # attention_mask.shape

    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    # mask.shape

    # 上面的每个向量表示一个单独token的掩码现在每个token都有一个大小为768的向量，表示它的attention_mask状态。然后将两个张量相乘:

    masked_embeddings = embeddings * mask
    # masked_embeddings.shape

    # torch.Size([2, 128, 768])
    torch.Size([data_num, 128, 768])

    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask

    # print(mean_pooled)

    # print(type(mean_pooled))

    return mean_pooled

# sentences = [
#    "你叫什么名字？",
#    "你的名字是什么？",
#    "你的名字是什么？",
#    "你的名字是什么？",
#    "你的名字是什么？",
#    "你的名字是什么？",
#    "你的名字是什么？",
#    "你的名字是什么？",
#    "你的名字是什么？",
# ]
# sb = split_batch(sentences, 2)
# embs = []
# for batch in sb:
#	emb = encode(batch)
#	embs += emb
#
# print(embs)
# print(len(embs))




