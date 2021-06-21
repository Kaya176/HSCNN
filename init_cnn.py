import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchtext import vocab
from torchtext.data.field import LabelField
from torchtext.data.iterator import pool
import math
from load_data import load_data
import numpy as np
from sklearn.metrics import accuracy_score
class CNN_network(nn.Module):

    def __init__(self,embedding,output_size,in_channels,out_channels,kernel_size,stride,padding,embedding_size,keep_probab,d):

        super(CNN_network,self).__init__()

        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        #self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.keep_prob = keep_probab
        self.d = d

        self.word_embedding = embedding
        self.conv1 = nn.Conv2d(in_channels,out_channels,(kernel_size[0],embedding_size),stride,padding)
        self.conv2 = nn.Conv2d(in_channels,out_channels,(kernel_size[1],embedding_size),stride,padding)
        self.conv3 = nn.Conv2d(in_channels,out_channels,(kernel_size[2],embedding_size),stride,padding)

        self.dropout = nn.Dropout(keep_probab)
        self.linear = nn.Linear(len(kernel_size)*out_channels,1024)
        self.label = nn.Linear(1024,22)
        self.sig = nn.Sigmoid()
        self.out = nn.Linear(1024,output_size)
        self.linear_onehot = nn.Linear(22,1024)
        self.relu = nn.ReLU()

    def conv_pool(self,input,conv_layer):
        conv_out = conv_layer(input)
        activation = F.relu(conv_out.squeeze(3))
        pool_out = F.max_pool1d(activation,activation.size()[2]).squeeze(2)
        return pool_out
        
    def forward(self,input_sent):
        '''
        input_sent : sentences
        label_onehot : one-hot vector로 변환된 라벨(ex : [0,0,0,0,1] 등)
        '''
        input_sentence = input_sent.transpose(1,0)
        #print(input_sentence)
        input = self.word_embedding(input_sentence)
        input = self.dropout(input)
        input = input.unsqueeze(1)

        pool_out1 = self.conv_pool(input,self.conv1)
        pool_out2 = self.conv_pool(input,self.conv2)
        pool_out3 = self.conv_pool(input,self.conv3)
        all_pool_out = torch.cat([pool_out1,pool_out2,pool_out3],dim = 1)
        x = self.dropout(all_pool_out)
        #self.dropout1 = nn.Dropout(0.25)

        x = self.linear(x)
        cnn_out = self.label(x)

        #q_w = self.linear_onehot(label_onehot)
        #q_w = self.relu(q_w/math.sqrt(self.d))

        return x,cnn_out

train_data,test_data,embedding_layer = load_data()
kernel_size = [3,4,5]
keep_prob = 0.25
embedding_size = 300
d = 22
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CNN_network(embedding = embedding_layer,output_size = 1,in_channels= 1,out_channels=128,kernel_size=kernel_size,stride=1,padding = False,keep_probab=keep_prob,
                    embedding_size=embedding_size,d=d).to(device)

optimizer = optim.Adam(model.parameters(),lr = 1e-3)

def make_onehot(labels):
    result = np.zeros(22) #label의 총 갯수
    labels = labels.split(sep = " ")
    for la in labels:
        num = int(la[-2:])
        result[num-1] = 1
    return result

def train(train_data,model):
    #define loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train() #train mode
    train_loss = 0
    count = 0 
    epochs = 100
    for epoch in range(epochs):
        
        for batch_idx,batch in enumerate(train_data):
            text = batch.text
            label = batch.label
            #label = label.unsqueeze(1)
            #label = label.clone().detach().requires_grad_(True)

            text.to(device)
            label.to(device)
            
            optimizer.zero_grad()
            x,out = model(text)
            loss = loss_fn(out,label)
            count += 1
            train_loss += loss
            loss.backward()
            optimizer.step() #가중치 수정
        print(f"Epoch : {epoch} \t Train  Loss : {train_loss/count : .3f}")

train(train_data,model)
#save model
state = {
    "state_dict" : model.state_dict(),
    'optimizer' : optimizer.state_dict()
}
torch.save(state,"CNN-checkpoint.pt")