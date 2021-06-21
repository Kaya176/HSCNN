import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HSCNN(nn.Module):

    def __init__(self,embedding_layer,output_size,in_channels,out_channels,kernel_size,stride,padding,embedding_size,dropout_rate,c):

        super(HSCNN,self).__init__()

        self.c = c # the number of training instances of a category

        #이하 부분은 CNN 구현에 필요한 부분.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size =  kernel_size
        self.stride = stride
        self.padding = padding

        #layers
        self.embedding_layer = embedding_layer #init값으로 embedding_layer 자체를 받음.
        self.conv1 = nn.Conv2d(in_channels,out_channels,(kernel_size[0],embedding_size),stride,padding)
        self.conv2 = nn.Conv2d(in_channels,out_channels,(kernel_size[1],embedding_size),stride,padding)
        self.conv3 = nn.Conv2d(in_channels,out_channels,(kernel_size[2],embedding_size),stride,padding)

        self.dropout = nn.Dropout(dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(len(kernel_size)*out_channels,1024)
        self.label = nn.Linear(1024,22)
        self.sig = nn.Sigmoid()
        self.out = nn.Linear(1024,output_size)
        self.linear_onehot = nn.Linear(22,1024)
        self.relu = nn.ReLU()

    def conv_pool(self, input, conv_layer):
        conv_out = conv_layer(input)
        activation = F.relu(conv_out.squeeze(3))
        pool_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)
        return pool_out

    def forward_one(self, input_sentences, onehot):

        input_sentences = input_sentences.transpose(1, 0)
        input = self.embedding_layer(input_sentences)
        input = self.dropout1(input)
        input = input.unsqueeze(1)

        pool_out1 = self.conv_pool(input, self.conv1)
        pool_out2 = self.conv_pool(input, self.conv2)
        pool_out3 = self.conv_pool(input, self.conv3)
        all_pool_out = torch.cat([pool_out1, pool_out2, pool_out3], dim = 1)
        x = self.dropout(all_pool_out)

        x = self.linear(x)
        cnn_out = self.label(x)

        q_w = self.linear_onehot(onehot)
        q_w = self.relu(q_w / math.sqrt(self.c)) #h_c

        return x, q_w, cnn_out

    def forward_sia(self, input1, input2, onehot1, onehot2, state1):

        if state1 == 'train':
            x1, q_w1, cnn_out1 = self.forward_one(input1, onehot1)
            out1 = self.sig(x1)
            x2, q_w2, cnn_out2 = self.forward_one(input2, onehot2)
            out2 = self.sig(x2)

            dis = torch.abs(out1 - out2)
            tmp = torch.mul(dis, q_w2)
            out = self.out(tmp)

            return out1, out2, cnn_out1, cnn_out2, out

        if state1 == 'prediction':
            out1 = input1
            out2 = input2

            dis = torch.abs(out1 - out2)
            q_w2 = self.liner_onehot(onehot2)
            tmp = torch.mul(dis, q_w2)
            out = self.out(tmp)

            return out1, out2, out