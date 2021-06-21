#load data part
'''
TabularDataset : Defines a Dataset of columns stored in CSV, TSV, or JSON format.
BucketIterator : Defines an iterator that batches examples of similar lengths together.
fasttext embedding을 사용하기 위해서는 pre-train 모델과 내가 직접 데이터로 훈련한 model 둘 다 필요한데,
pre-trained model : wiki.en.vec 을사용하고,
fine-tuned model은 저장을 vec 파일로 한 뒤, torchtext에서 불러와야 하는것으로 보인다.
-> gensim에서 save하는 함수는 bin파일로 저장하는 형태

나중에 알아보도록 하고, 우선 pre-train된 파일을 이용하여 진행하는것으로 하고,
밤에 다른 방법을 이용하여 fine-tuned된 파일을 torchtext에서 사용하는 방법을 이용해보자.
'''
from torch._C import device
from torch.nn.functional import embedding
from torch.nn.modules.container import Sequential
import torchtext
from torchtext import data
import torch
import torch.nn as nn
import numpy as np

def make_onehot(labels):
    result = np.zeros(22) #label의 총 갯수
    labels = labels.split(sep = " ")
    for la in labels:
        num = int(la[-2:])
        result[num-1] = 1
    return result

def load_data():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    TEXT = data.Field(fix_length = 500)
    LABEL = data.Field(sequential =False,is_target = True, use_vocab = False,dtype = torch.float64)
    #ONEHOT = data.Field(Sequential = False, is_target = True, use_vocab = False, dtype = torch.float64)

    field = {'text': ('text', TEXT),'label': ('label', LABEL)}
    field1 = [('text',TEXT),("label",LABEL)]
    train_pairs,test_pairs = data.TabularDataset.splits(
        path = '.',
        train='tmc2007-train.csv', test = 'tmc2007-test.csv',
        format='csv',
        fields=field1,
        skip_header = True
    )
    for i in range(len(train_pairs)):
        idx = vars(train_pairs[i])['label']
        idx = make_onehot(idx)
        vars(train_pairs[i])['label'] = idx

    for i in range(len(test_pairs)):
        idx = vars(test_pairs[i])['label']
        idx = make_onehot(idx)
        vars(test_pairs[i])['label'] = idx

    #Embedding vectors - Pretrained
    vector = torchtext.vocab.Vectors(name = 'wiki.en.vec')

    #Embedding vecotrs - Custom vector
    #vector = torchtext.vocab.Vectors(name = "MyCustomEmbeddingVectors") 

    TEXT.build_vocab(train_pairs,vectors = vector)

    #print("length of Text Vocab : ",len(TEXT.vocab))
    #print("Dim of Text : ",TEXT.vocab.vectors.size()[1])

    train_pair_batch = data.BucketIterator(
        dataset = train_pairs,
        sort = False,
        batch_size= 5,
        repeat = False,
        shuffle= True,
        device = device
    )
    
    test_pair_batch = data.BucketIterator(
        dataset = test_pairs,
        sort = False,
        batch_size= 5,
        repeat = False,
        shuffle= True,
        device = device
    )
    
    embedding_layer = nn.Embedding.from_pretrained(TEXT.vocab.vectors,freeze= False)
    #print(list(TEXT.vocab.stoi.keys())[10])
    #print(embedding_layer(torch.LongTensor([10,20,30]))) #'at'에 해당하는 embedding vector값
    #print()
    return train_pair_batch,test_pair_batch,embedding_layer

if __name__ == "__main__":
    train,test,layer = load_data()
    print(f"훈련 샘플의 갯수 : {len(train)}")
    print(f"테스트 샘플의 갯수 : {len(test)}")
    
    print("-"*50)
    print("샘플 뽑아보기")
    print("[train sample]")
    batch = next(iter(test))
    print(batch.text)
    print("-"*50)
    print("[test sample]")
    print(vars(test[10]))