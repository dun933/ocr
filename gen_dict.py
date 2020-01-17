# import re
# a = '<font color=red>腾讯科技</font>（<font color=red>深圳</font>）<font color=red>有限公司</font>'
#
# test = re.sub('<[\d\D]+?>','',a)
# print(test)

import os
import pickle
import random

with open('data/corpus/sougou_all.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()

lines = []
for line in data:
    line_striped = line.strip()
    line_striped = line_striped.replace('\u3000', '')
    line_striped = line_striped.replace('&nbsp', '')
    line_striped = line_striped.replace("\00", "")

    if line_striped != u'' and len(line.strip()) > 1:
        lines.append(line_striped)

# 所有行合并成一行
split_chars = [',', '，', '：', '-', ' ', ';', '。']
splitchar = random.choice(split_chars)
whole_line = splitchar.join(lines)
text = whole_line
str_list = [char+text[i+1] for i,char in enumerate(text[:-1])]
# print(str_list)
dict_count = {}
str_count = {}
for i in text:
    if i in dict_count.keys():
        dict_count[i] += 1
    else:
        dict_count[i] = 1
for i in str_list:
    if i in str_count.keys():
        str_count[i] += 1
    else:
        str_count[i] = 1
for k,v in str_count.items():
    # print(k,v)
    str_count[k] = v / dict_count[k[0]]
# for k,v in str_count.items():
#     print(k,v)
# print(str_count)
with open('pkl_file.pkl','wb') as f:
    pickle.dump(str_count,f)