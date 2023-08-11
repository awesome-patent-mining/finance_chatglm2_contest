import json
import pandas as pd
import glob

"""
整体逻辑：
## 1.从table2string_dir中获得表格转化的字符串，
## 2.将这些字符串放入dataframe，将500qie.csv中存储的字符串导入dataframe，将这两个dataframe合并
## 3.大dataframe中内容存入combine_pdf_text_table_info.csv
"""

question_list = []
file_csv_path = '500qie.csv'
file_csv_df = pd.read_csv(file_csv_path)

table_2_string_list = []
file_names = glob.glob('table2string_dir/*')
file_names = sorted(file_names, reverse=True)
for idx,f_abs_name in enumerate(file_names):
    print(idx,f_abs_name)
    f_short_name = f_abs_name.split('/')[-1]
    f_short_name = '__'.join(f_short_name.split('__')[1:])
    f_short_name = f_short_name.replace('__年度报告_txt.txt','')
    f_short_name = f_short_name.replace('__','_')
    with open(f_abs_name,'r') as f:
        lines = f.readlines()
        for idx,line in enumerate(lines):
            table_2_string_list.append([f_short_name,idx,line])

table_2_string_df = pd.DataFrame(table_2_string_list,columns=['报告名','行ID','文本'])
combine_text_table_string_df = pd.concat([file_csv_df,table_2_string_df])

combine_text_table_string_df.to_csv('combine_pdf_text_table_info.csv')