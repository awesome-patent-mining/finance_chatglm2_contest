import json
import pandas as pd
import glob
import time
from langchain.schema import Document
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import sentence_transformers
from tqdm import tqdm
from multiprocessing import Process, cpu_count, Pool
"""
整体逻辑：
## 1.将与5000个问题相关的pdf找出来，存入‘ques_pdf_keyword_map_raw.txt’
##   格式为‘第2个问题：2019年四方科技电子信箱是什么?,搜索结果中对应的文件名为：四方科技集团股份有限公司(简称四方科技)2019年发布了金融报告’，
## 2.将combine_pdf_text_table_info.csv中所包含的5000个pdf的相关内容取出
## 3.使用多进程并发，将这5000个文件中的内容转化为Faiss文件，存放在faiss_local_file_combine_text_table 
"""
start1 = time.time()
question_list = []
file_csv_path = 'combine_pdf_text_table_info.csv'
file_csv_df = pd.read_csv(file_csv_path)
file_df_groupby_filename = file_csv_df.groupby('报告名')['文本'].apply(list)
end1 = time.time()
cost1 = end1 - start1
print("数据加载完毕！用时：{}".format(cost1))

ques_pdf_keyword_map_list = []
with open('ques_pdf_keyword_map_raw.txt','r') as f:
    lines = f.readlines()
    for line in lines:
        begin_pos = line.find('(简称')
        end_pos = line.find(')', begin_pos)
        short_name = line[begin_pos+3:end_pos]
        release_year = line[end_pos+1:end_pos+5]
        ques_pdf_keyword_map_list.append(short_name+'_'+release_year+'年')
print("目标索引加载完毕！现在开始进行t2v")

def doc2vec(docid):
    doc = docid[0]
    id = docid[1]
    start = time.time()
    embeddings = HuggingFaceEmbeddings(model_name="text2vec/text2vec-large-chinese",)
    embeddings.client = sentence_transformers.SentenceTransformer(embeddings.model_name, device="cuda:1")
    vector_store = FAISS.from_documents(doc, embeddings)
    del embeddings
    vector_store.save_local('faiss_local_file_combine_text_table/' + id)
    del vector_store
    end = time.time()
    cost = end - start
    print(id + '-向量保存完毕！用时：{}'.format(cost))

vec_start = time.time()
pool = Pool(20)
for index,idx in enumerate(file_df_groupby_filename.index):
    docs = []
    print(index,idx)
    idx_split_list = idx.split('_')
    new_idx = '_'.join(idx_split_list[-2:])
    if ques_pdf_keyword_map_list.count(new_idx)>0:
        print(f'create FAISS index of {new_idx}')
        for idx_1,row in enumerate(file_df_groupby_filename[idx]):
            metadata = {"source": f'doc_id_{idx}'}
            docs.append(Document(page_content=row, metadata=metadata))
        pool.map_async(doc2vec, ([docs, new_idx], ))
        # vector_store = FAISS.from_documents(docs, embeddings)
        # vector_store.save_local('faiss_local_file/'+new_idx)
pool.close()  
pool.join()
vec_end = time.time()
vec_time = vec_end - vec_start
print(f"哈哈哈{vec_time}")