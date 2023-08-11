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

## 1.加载chatglm-2，
nltk.data.path = [os.path.join(os.path.dirname(__file__), "nltk_data")] + nltk.data.path
tokenizer = AutoTokenizer.from_pretrained("model_2_huggingface", trust_remote_code=True)
config = AutoConfig.from_pretrained("model_2_huggingface", trust_remote_code=True)
config.pre_seq_len = 128

model = AutoModel.from_pretrained("model_2_huggingface", config=config, trust_remote_code=True).half().cuda()
#prefix_state_dict = torch.load(os.path.join("output/checkpoint-900", "pytorch_model.bin"))
#new_prefix_state_dict = {}
#for k, v in prefix_state_dict.items():
#    new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
#model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

model = model.eval()

## 2.加载text2vec/text2vec-large-chinese
embeddings = HuggingFaceEmbeddings(
         model_name="text2vec/text2vec-large-chinese")

"""Override Chatbot.postprocess"""
def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y
def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text
# 先获取已经存在的FAISS索引名称
FAISS_fold_path = './faiss_local_file_combine_text_table'
max_length = 20000
top_p = 5
temperature = 0.01
file_names = glob.glob(FAISS_fold_path + '/*')
file_names = sorted(file_names, reverse=True)
FAISS_index_list = []
for fname in file_names:
    fname = fname.split('/')[-1]
    FAISS_index_list.append(fname)
# 获取问题和答案所在pdf名称的对应关系
ques_pdf_keyword_map_list = []
with open('good-5.txt','r') as f:
    lines = f.readlines()
    for line in lines:
        begin_pos = line.find('(简称')
        end_pos = line.find(')', begin_pos)
        short_name = line[begin_pos+3:end_pos]
        release_year = line[end_pos+1:end_pos+5]
        ques_pdf_keyword_map_list.append(short_name+'_'+release_year+'年')
# 循环5000个问题，如果这个问题对应的pdf已经生成了FAISS索引，将这个索引调出，从中找出top_k个最可能的句子，然后一起放到chatGLM2中，来生成答案
#question_list = []
file_csv_path = './chatglm_llm_fintech_raw_dataset/5000_questions_multi_ques.csv'
questions_multi_type_df = pd.read_csv(file_csv_path)
question_list = list(questions_multi_type_df['无企业名称和年份的正文'])
raw_question_list = list(questions_multi_type_df['正文'])
corp_list = list(questions_multi_type_df['机构'])
year_list = list(questions_multi_type_df['时间'])

result_list = []
## 3.对5000个问题循环执行


for idx,ques in enumerate(question_list):
    ## 4.对于每个问题，从ques_pdf_keyword_map_raw.txt中找到对应的pdf文件
    pdf_keywords = ques_pdf_keyword_map_list[idx]
    if FAISS_index_list.count(pdf_keywords)>0:
        data_tmp = dict()
        data_tmp['id'] = idx
        ## 4.对这个问题，从5000_questions_multi_ques.csv中找到该问题去掉企业名称后的版本，比如 四方科技的电子邮箱是多少？--》邮箱是多少
        data_tmp['question'] = ques
        vecdb = FAISS.load_local(FAISS_fold_path+"/"+pdf_keywords, embeddings)
        ## 5.拿着去掉企业名称的问题，到pdf对应的faiss文件中检索相关段落/句子
        proxy_chain = init_chain_proxy(ProxyLLM(), vecdb, top_p)
        q = proxy_chain.run(ques)
        q = parse_text(q)
        ## 5.拿着没有去掉企业名称的问题，连同faiss中检索到的相关句子，组成prompt

        prompt_template = """请你基于已知信息回答用户的问题：""" + raw_question_list[idx] + """\n已知信息:我们从""" + corp_list[idx] + """公司""" + year_list[idx] + """的金融报告的表格中抽取以下内容："""+q




        #proxy_chain = init_chain_proxy_1(ProxyLLM(), vecdb,raw_question_list[idx],corp_list[idx],year_list[idx], top_p)

        #q = proxy_chain.run(ques)
        '''
        while len(q)>=max_length:
            top_tmp = top_p-1
            proxy_chain = init_chain_proxy_1(ProxyLLM(), vecdb, top_tmp)
            q = proxy_chain.run(ques)
        '''

        history = []
        #print(f'the length of prompt for question {idx} is {len(q)}')
        ## 6.将promt放入大模型，获得结果
        response,history = model.chat(tokenizer, prompt_template, history, max_length=max_length, top_p=top_p,temperature=temperature)

        # history = answer['history']
        print(f'第{idx}个问题是：{raw_question_list[idx]}，chatglm2的回答是:{response}')
        #print(f'第{idx}个问题是：{ques},该问题对应的pdf关键词是:{pdf_keywords},prompt是：{prompt_template}')
        data_tmp['answer'] = response
        result_list.append(data_tmp)
    else:
        pass
        #print(f'第{idx}个问题是：{ques},该问题对应的pdf关键词是{pdf_keywords}，目前相关pdf文件内容还没有入FAISS')
with open('model_2_raw_huggingface.jsonl','w') as f:
    for result_tmp in result_list:
        result_json = json.dumps(result_tmp,ensure_ascii=False)
        f.write(result_json+'\n')


