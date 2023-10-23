import numpy as np
import torch 
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel

model_name = 'chinese-roberta-wwm-ext'
MODEL_PATH = '/home/wangyp/pytorch_chinese_lm_pretrain-master/output/'

# a. 通过词典导入分词器
tokenizer = BertTokenizer.from_pretrained(model_name)
# b. 导入配置文件
model_config = BertConfig.from_pretrained(model_name)
# 修改配置
model_config.output_hidden_states = True
model_config.output_attentions = True
# 通过配置和路径导入模型
bert_model = BertModel.from_pretrained(MODEL_PATH, config = model_config)


text = "我是小辣椒"
input_ids = tokenizer.encode(text, add_special_tokens=True)
input_ids = torch.tensor(input_ids).unsqueeze(0)  # 添加batch维度
# print(tokenizer.encode_plus('吾儿莫慌'))
# print(tokenizer.convert_ids_to_tokens(tokenizer.encode('吾儿莫慌')))

outputs = bert_model(input_ids)

# print(outputs)

hidden_states = outputs[0]  # 获取模型的所有隐藏状态
last_hidden_state = hidden_states[-1]  # 获取最后一层的隐藏状态

print(last_hidden_state[0].shape)
print(last_hidden_state[0])


