"""
整体逻辑：
## 1.使用bert对表格进行分类，分类后以pdf文件为单位，将其中的表格存放在csv文件中，文件名称和pdf名称保持一致，
## 2.在csv文件中，每个表格记录包含四列，即pdf_id(文件名),table_id（表在文件中的序号）,label（表格类别，0为横，1为纵，2为组合表）,abstract（markdown格式的表格内容）
##   由于2类表格情况特殊，暂时没处理
## 3.将全部csv文件读入进来，然后合并成一个大dataframe
## 4.再以pdf_id为标准，groupby一下，获取每个pdf中的所有表格
## 5.对于每个表格，实现表格中的每行转化成一个句子，第1类表格，直接转化，第2类，将表头和表格cell内容对应上，实现转化
## 6.实现每个pdf中的表格全部转化为句子
## 7.把每个pdf中的句子整理为txt文档



"""
import glob
import pandas as pd
# 文件夹路径
#3.将全部csv文件读入进来，然后合并成一个大dataframe
folder_path = '/root/chenliang/finance_table_classification/table_type_classification_result'
# 获取文件夹内所有文件名称
file_names = glob.glob(folder_path + '/*')
file_names = sorted(file_names, reverse=True)

df_list = []
for abs_path in file_names:
    test_All_tmp = pd.read_csv(abs_path)
    df_list.append(test_All_tmp)
all_csv_df = pd.concat(df_list)
#4.再以pdf_id为标准，groupby一下，获取每个pdf中的所有表格
file_df_groupby_filename = all_csv_df.groupby('pdf_id')

# 5.对于每个表格，实现表格中的每行转化成一个句子，第1类表格，直接转化，第2类，将表头和表格cell内容对应上，实现转化
table_sentence_list_all = []
for idx_0,row_0 in file_df_groupby_filename:
    print(idx_0)
    table_sentence_list_per_pdf = []
    for idx_1,row_1 in row_0.iterrows():
        #rint(idx_1)
        #第1类表格，直接转化，
        if row_1['label']==0:
            for table_row_tmp in row_1['abstract'].split('\n'):
                span_list_tmp = table_row_tmp.replace('|','')
                table_sentence_list_per_pdf.append(span_list_tmp)
        #第2类，将表头和表格cell内容对应上，实现转化
        elif row_1['label'] ==1:
            row_list_tmp = row_1['abstract'].split('\n')
            row_head = row_list_tmp[0].split('|')
            for row_2 in row_list_tmp[1:]:
                if row_2!='':
                    row_content = row_2.split('|')
                    #rint(row_2)
                    #rint(row_content)
                    span_list_tmp = ''
                    for head_idx,each_head in enumerate(row_head):
                        try:
                            if each_head!='':
                                if row_content[head_idx]=='':
                                    pass
                                else:
                                    span_list_tmp= span_list_tmp+','+each_head+'是'+row_content[head_idx]
                            else:
                                if row_content[head_idx]=='':
                                    pass
                                else:
                                    span_list_tmp= span_list_tmp+','+row_content[head_idx]
                        except Exception as e:
                            pass
                            #rint(e)
                    table_sentence_list_per_pdf.append(span_list_tmp)
    table_sentence_list_all.append([idx_0,table_sentence_list_per_pdf])

    for pdf_id, abstract in table_sentence_list_all:
        print(pdf_id)
        with open('table2string_dir/' + pdf_id, 'w') as f:
            for sentence in abstract:
                if sentence != '':
                    f.write(sentence + '\n')