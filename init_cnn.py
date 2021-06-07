import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data.iterator import pool

class CNN_network(nn.Module):

    def __init__(self,output_size,in_channels,out_channels,kernel_size,stride,padding,vocab_size,embedding_size,keep_probab,d):

        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.keep_prob = keep_probab
        self.d = d

        self.word_embedding = nn.Embedding(vocab_size,embedding_size)
        self.conv1 = nn.Conv2d(in_channels,out_channels,(kernel_size[0],embedding_size),stride,padding)
        self.conv1 = nn.Conv2d(in_channels,out_channels,(kernel_size[1],embedding_size),stride,padding)
        self.conv1 = nn.Conv2d(in_channels,out_channels,(kernel_size[2],embedding_size),stride,padding)

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
        pool_out = F.max_pool1d(activation,activation.sizes()[2]).squeeze(2)
        return pool_out
        
    def forward(self,input_sent,label_onehot):
        '''
        input_sent : sentences
        label_onehot : one-hot vector로 변환된 라벨(ex : [0,0,0,0,1] 등)
        '''

if __name__ == '__main__':
    model  = CNN_network(???)
    #구현해야할것 : train & checkpoint