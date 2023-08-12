from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
import gradio as gr
import mdtex2html
import torch
import os
import json
import sys
import re
import glob
from langchain.vectorstores import FAISS
from ptuning.arguments import ModelArguments, DataTrainingArguments

import sentence_transformers
"""
整体逻辑：
## 1.加载chatglm-2，
## 2.加载text2vec/text2vec-large-chinese
## 3.对5000个问题循环执行
## 4.对于每个问题，从ques_pdf_keyword_map_raw.txt中找到对应的pdf文件
## 4.对这个问题，从5000_questions_multi_ques.csv中找到该问题去掉企业名称后的版本，比如 四方科技的电子邮箱是多少？--》邮箱是多少
## 5.拿着去掉企业名称的问题，到pdf对应的faiss文件中检索相关段落/句子
## 5.拿着没有去掉企业名称的问题，连同faiss中检索到的相关句子，组成prompt
## 6.将promt放入大模型，获得结果
"""
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from utils import ProxyLLM, init_chain_proxy_1, init_chain_proxy,init_knowledge_vector_store
import nltk
import json
import pandas as pd
import glob


file_csv_path = './chatglm_llm_fintech_raw_dataset/5000_questions_multi_ques.csv'
questions_multi_type_df = pd.read_csv(file_csv_path)
question_list = list(questions_multi_type_df['无企业名称和年份的正文'])
raw_question_list = list(questions_multi_type_df['正文'])
corp_list = list(questions_multi_type_df['机构'])
year_list = list(questions_multi_type_df['时间'])

result_list = []
## 3.对5000个问题循环执行
result_dir = './result_20230808'


file_names = glob.glob(result_dir + '/*')
file_names = sorted(file_names, reverse=True)
result_index_list = []
for fname in file_names:
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for result_tmp in lines:
            result_json = json.loads(result_tmp)
            result_index_list.append(result_json)

id_ques_answer_dict = dict()
for result_tmp in result_index_list:
    ques_id = result_tmp['id']
    id_ques_answer_dict[ques_id] = {'question': raw_question_list[ques_id], 'answer': result_tmp['answer']}

with open(f'combined_result_20230808.json', 'w', encoding='utf-8') as f:
    for idx,ques in enumerate(raw_question_list):
        if id_ques_answer_dict.get(idx,-1)==-1:

            result_tmp = {'id':idx,'question':ques,'answer':''}
            print(result_tmp)
        else:
            result_tmp = {'id': idx, 'question': id_ques_answer_dict[idx]['question'], 'answer': id_ques_answer_dict[idx]['answer']}
        result_json = json.dumps(result_tmp, ensure_ascii=False)
        f.write(result_json + '\n')








