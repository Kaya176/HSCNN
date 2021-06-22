'''
각 카테고리별에 속한 문장들은 유사한 문장으로 설정하고
다른 카테고리에 속한 문장들은 다른 문장이라고 판단하기
=> random이긴 함..ㅇㅇ
비율은 1:2 ? 1:1? -> 우선 1:2로 설정.
'''

import pandas as pd
from load_data import make_onehot
import collections
import random

def make_pairs(file_name):
    file = pd.read_csv(file_name)

    #set default dictionary : key -> labels
    result = collections.defaultdict(list)
    key = [str(i) if len(str(i))>1 else "0"+str(i) for i in range(1,23)]
    for k in key:
        result[k]

    #split each labels 
    for idx in range(len(file)):
        for label in file.iloc[idx]['label'].split():
            result[label].append(idx)
    
    pair_data_sim = []
    pair_data_dissim = []
    #pairing
    for l in result.keys():
        #similar part
        sample = random.sample(result[l],300)
        for i in range(150):
            if i == 149:
                pair = sample[i:]
            else:
                pair = sample[2*i:2*i+2]
            pair.append(l)
            pair_data_sim.append(pair)
        #dissimilar part
        temp = list(result.keys())[:]
        temp.remove(l)
        except_label = temp[:]
        #print(except_label)
        for i in range(300):
            except_index = random.choice(except_label)
            p_1 = random.choice(result[l])
            p_2 = random.choice(result[except_index])
            pair = [p_1,p_2,l,except_index]
            pair_data_dissim.append(pair)
    #output file
    t_1 = []
    t_2 = []
    labels_1 = []
    labels_2 = []
    for i in pair_data_sim:
        t_1.append(file.iloc[i[0]]['text'])
        t_2.append(file.iloc[i[1]]['text'])
        labels_1.append(file.iloc[i[0]]['label'])
        labels_2.append(file.iloc[i[1]]['label'])
    for i in pair_data_dissim:
        t_1.append(file.iloc[i[0]]['text'])
        t_2.append(file.iloc[i[1]]['text'])
        labels_1.append(file.iloc[i[0]]['label'])
        labels_2.append(file.iloc[i[1]]['label'])
    dataframe = pd.DataFrame({'text1':t_1,"text2" : t_2,'label':labels_1,'onehot':labels_2})
    return pair_data_sim,pair_data_dissim,dataframe


if __name__ == '__main__':
    sim,dissim,frame = make_pairs('tmc2007-train.csv')
    print("Simiar pair data...")
    print(sim[:10])
    print("-"*100)
    print("Dissimilar pair data...")
    print(dissim[:10])
    print("make datafile...")
    frame.to_csv("tmc2007-pair-train.csv",index= False)
    print("Done!")