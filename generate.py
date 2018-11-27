#!/usr/bin/env python
# coding: utf-8

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

class LangModel(nn.Module):
    
    def __init__(self, embedder, rnns, decoder):
        super(LangModel, self).__init__()
        self.embedder = embedder
        self.rnns = rnns
        self.decoder = decoder
        
    def forward(self,input):
        out=embedder(input)
        for rnn in rnns:
            out,hid = rnn(out)
        out = decoder(out[:,-1])
        return out

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def generate(length,creativity,dist=False):
    next_word = "."
    text = ""
    for i in range (length):
        tensor_output = model(torch.tensor([[stoi[next_word]]],dtype=torch.long,device="cpu"))[0]
        output = (tensor_output).detach().cpu().numpy()
        
#hard cutoff
#         subdist = softmax(np.sort(output)[-creativity:])
#         print (subdist)
#         next_word = itos2[np.random.choice(np.argsort(output)[-creativity:],p=subdist)]

#soft cutoff
        distribution = softmax(output/creativity)
        ranks = np.argsort(distribution)
        if dist:
            print([itos[rank] for rank in ranks[-10:]])
            distribution.sort()
            print (distribution[-10:])
            break
        next_word = itos[np.random.choice(range(distribution.shape[0]),p=distribution)]
        text+=(next_word+" ")
    return text

parser = argparse.ArgumentParser(description='Generate a CreepyPasta story.')
parser.add_argument('--length', metavar='WORDS', type=int, help='how many words?', default=1000)
parser.add_argument('--paragraph_mean_length', metavar='WORDS_IN_PAR', type=int, help='roughly how long should a paragraph be?', default=100)
parser.add_argument('--paragraph_standard_deviation', metavar='STD_DEV_PAR_LENGTH', type=float, help='how much should paragraph length vary? (between 0 and 1)', default=0.5)
parser.add_argument('--start_crazy', metavar='START_CRAZY', type=float, help='how crazy should it start?', default=0.5)
parser.add_argument('--end_crazy', metavar='END_CRAZY', type=float, help='how crazy should it end?', default=1.0)
parser.add_argument('--weights_dir', metavar='WEIGHTS_DIR', type=str, help='where are the weights?', default="creepy_pasta_weights")
parser.add_argument('--verbose', metavar="VERBOSE",type=bool,help="print the story to the console (it's always written to mynovel.txt)",default=True)

args = parser.parse_args()

# Import the weights
WEIGHTS_DIR = args.weights_dir+"/"
with open(WEIGHTS_DIR+"itos.pkl",'rb') as f:
    itos=pickle.load(f)
with open(WEIGHTS_DIR+"stoi.pkl",'rb') as f:
    stoi=pickle.load(f)
generator_state_dict = torch.load(WEIGHTS_DIR+"generator.weights",map_location=lambda storage, loc: storage)
    
vocab_size = len(itos)
embedding_size = 400


# Construct the model
rnn0 = nn.LSTM(400, 1150, 1)

rnn1 = nn.LSTM(1150, 1150, 1)

rnn2 = nn.LSTM(1150, 400, 1)

rnns = nn.ModuleList([rnn0,rnn1,rnn2])

embedder= nn.Embedding(vocab_size,embedding_size)
decoder = nn.Linear(embedding_size,vocab_size,bias=False)

model = LangModel(embedder,rnns,decoder)

model.load_state_dict(generator_state_dict)
        
expected_length = args.length
doc_length = 0
start_crazy = args.start_crazy
end_crazy = args.end_crazy

with open("mynovel.txt","w") as f:
    crazification_rate = (end_crazy-start_crazy)/expected_length
    while doc_length < expected_length:
        par_length = max(1,int(np.random.normal(loc=args.paragraph_mean_length,
                                             scale=args.paragraph_mean_length*args.paragraph_standard_deviation,
                                             size=None)))
        crazy = start_crazy+doc_length*crazification_rate
        paragraph = generate(par_length,crazy)
        f.write(paragraph+"\n\n")
        if args.verbose:
            print(paragraph)
        doc_length += par_length