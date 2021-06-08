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
from torch.nn.modules.container import Sequential
import torchtext
from torchtext import data
import torch

def make_onehot(labels):
    
def load_data():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    TEXT = data.Field(fix_length = 500)
    LABEL = data.Field(Sequential =False,is_target = True,use_vobcab = False,dtype = torch.float64)
    #ONEHOT = data.Field(Sequential = False, is_target = True, use_vocab = False, dtype = torch.float64)

    field = {'text': ('text', TEXT),'label': ('label', LABEL)}

    train_pairs= data.TabularDataset.splits(
        train='tmc2007-train.csv',
        format='csv',
        fields=field,
        skip_header = True
    )
    test_pairs = data.TabularDataset.split(
        test = 'tmc2007-test.csv',
        format = 'csv',
        fileds = field,
        skip_header = True
    )

    #Embedding vectors - Pretrained
    vector = torchtext.vocab.Vectors(name = 'wiki.en.vec')
    #Embedding vecotrs - Custom vector
    #vector = torchtext.vocab.Vectors(name = "MyCustomEmbeddingVectors") 

    TEXT.build_vocab(train_pairs,vectors = vector)

    print("length of Text Vocab : ",len(TEXT.vocab))
    print("Dim of Text : ",TEXT.vocab.vectors.size()[1])

    train_pair_batch = data.BucketIterator(
        dataset = train_pairs,
        sort = False,
        batch_size= 100,
        repeat = False,
        shuffle= True,
        device = device
    )

    test_pair_batch = data.BucketIterator(
        dataset = train_pairs,
        sort = False,
        batch_size= 100,
        repeat = False,
        shuffle= True,
        device = device
    )

    return train_pair_batch,test_pair_batch