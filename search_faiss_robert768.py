import pickle
from faiss_index import faissIndex
import pandas as pd
import numpy as np
# from sentence_transformers import SentenceTransformer
# Download model
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2/')
from sentence_emb import encode


from transformers import BertTokenizer, BertModel
import torch
# First we initialize our model and tokenizer:
tokenizer = BertTokenizer.from_pretrained('./result')
model = BertModel.from_pretrained('./result').cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from flask import Flask, request, jsonify
# define flask app
app = Flask(__name__)


# faiss_index_path = "faiss_index384.pkl"
faiss_index_path = "faiss_index_robert.pkl"

symptom_name_df = pd.read_csv("col2.csv")

# 从本地加载faiss_index模型
def load_faiss_index(var_faiss_model_path):
    # 从本地加载faiss_index模型
    # with open('strategy/semantic_recall/model/tt.txt', 'r') as f:
    #     print(f.readlines())
    with open(var_faiss_model_path, mode='rb', errors=None) as fr:
        index = pickle.load(fr, encoding='ASCII', errors='ASCII')
        return index


def symptom_name_recall(symptom_name):
    # 将参数中当前的文本编码成向量
    sentence = []
    sentence.append(symptom_name)
    # qyery_emb = model.encode(sentence)
    qyery_emb = encode(sentence,tokenizer,model)
    # 去faiss中检索相近的faiss索引
    # 加载faiss
    loaded_faiss_index = load_faiss_index(faiss_index_path)
    # 寻找最近k个物料
    # R, D, I = loaded_faiss_index.search_items(qyery_emb.reshape([-1, 384]), k=10, n_probe=5)
    R, D, I = loaded_faiss_index.search_items(np.array(qyery_emb.reshape([-1, 768]).cpu()), k=10, n_probe=5)
    # 从faiss库中检索的物料ID进行转换
    result = []
    for id_list in R:
        for item in id_list:
            result.append(item)
    symptom_name_list = symptom_name_df[symptom_name_df['index'].isin(result)]['symptom_name'].to_list()

    # 从相似度检索的结果中，去除自己
    if symptom_name in symptom_name_list:
        symptom_name_list.remove(symptom_name)

    print(symptom_name + ' 的相近的词：' + str(symptom_name_list))
    return symptom_name_list


word_lsit = ['头痛','恶心吧吐','期饮酒','出血','失眠']
for word in word_lsit:
    symptom_name_recall(word)


@app.route("/similarity_search", methods=["GET","POST"])
def similarity_search():
    word = request.json.get("word")
    res = symptom_name_recall(word)
    return {"data": res}

if __name__ == '__main__':

    # 模型部署
    app.run(host='0.0.0.0', threaded=True, port=5013)
