### 代码说明：
***
**step_1_convert_table_to_string.py**:  将markdown格式的表格转化为文字
**step_2_combine_table_and_text.py**：将表格转化的文字和纯文本内容合并
**step_3_txt2vec.py**：利用text2vec-large-chinese将每个pdf中的文本信息转化为向量，存储在faiss中
**step_4_finance_generate_answer_４_model2_huggingface_1.py**：利用chatglm-2，回答 5000个问题
### 数据说明：
***
我们使用pdf-to-txt代码将11579个pdf转化为txt文件后，将每个txt拆分为两部分，第一部分是纯文本部分，我们以每行最多包含500个词汇的完整句子为标准，将纯文本分段，并存储在**500qie.csv**中， 每行包含三个字段，即pdf_id，句子顺序ID，句子内容；第二部分是表格字段，存储在**table1.zip**中，之后将这些表格转化为markdown格式，存储在**markdown1.zip**中  
**5000_questions.xlsx** 存放了5000个问题中抽取的企业名称和年份  
**good-5.txt**中存放着5000个问题对应的pdf文件