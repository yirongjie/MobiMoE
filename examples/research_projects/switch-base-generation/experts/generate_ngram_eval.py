def loadMoEData(file_name):
    fileHandler  =open(file_name,  "r")
    filelines = []
    while  True:
        line  =  fileHandler.readline()
        if  not  line  :
            break
        txt_ = line.strip().split("\n")[0]
        if len(txt_):
            filelines.append(txt_)
    fileHandler.close()
    seq_num = int(filelines[-1].split(" ")[0])
    seqs_en =[]
    seqs_de =[]
    for i in range(seq_num+1):
        seqs_en.append([])
        seqs_de.append([])
    for line in filelines:
        if line.split(" ")[1] == 'E':
            seqs_en[int(line.split(" ")[0])].append(line.split(" ")[2:])    
        if line.split(" ")[1] == 'D':
            seqs_de[int(line.split(" ")[0])].append(line.split(" ")[2:])
    result = []    
    for idx in range(0, len(seqs_en)):
        seq_en = seqs_en[idx]
        seq_de = seqs_de[idx]
        seqs_en_token =[]
        max_ = 0
        for m in seq_en:
            if int(m[0]) > max_:
                max_ = int(m[0])
        for i in range(max_+1):
            seqs_en_token.append([])
        seqs_de_token =[]
        max_ = 0
        for m in seq_de:
            if int(m[0]) > max_:
                max_ = int(m[0])
        for i in range(max_+1):
            seqs_de_token.append([])
        for et in seq_en:
            # print(seqs_en, et)
            seqs_en_token[int(et[0])].append(int(et[2]))
        for dt in seq_de:
            seqs_de_token[int(dt[0])].append(int(dt[2]))
        dict_ = {'E':seqs_en_token, 'D':seqs_de_token}
        result.append(dict_)
    return result

import os

def TrigramFormFile():
    experts_data = []
    data = loadMoEData(os.path.dirname(os.path.realpath(__file__))+"/st-b8-samsum.csv")
    seqLen = len(data)
    for seq_idx in range(seqLen):
        seq_d_tokens = data[seq_idx]['D'] # tokensize x 6
        for token_idx, seq_d_t_sparses in enumerate(seq_d_tokens):
            experts = ""
            for sparse_idx, seq_d_t_s_expert in enumerate(seq_d_t_sparses):
                experts = experts + str(seq_d_t_s_expert)
            # print(experts)
            experts_data.append(experts)
    return experts_data

listOfLines = TrigramFormFile()

# exit()

import torch
from torch import nn, optim

CONTEXT_SIZE = 2  # 2-gram
EMBEDDING_DIM = 10  # 词向量的维度

trigram = []
# # Open file    
# fileHandler = open(os.path.dirname(os.path.realpath(__file__))+"/test2.txt",  "r")
# # Get list of all lines in file
# listOfLines = fileHandler.readlines()
# Close file
# fileHandler.close()# Iterate over the lines
test_sentence = []
for line in listOfLines:
    tmp_test_sentence = list(line.strip())
    test_sentence = test_sentence+tmp_test_sentence
    for i in range(len(tmp_test_sentence)-2):
        trigram.append(((tmp_test_sentence[i], tmp_test_sentence[i+1]), tmp_test_sentence[i+2]))


print("CUDA support:", torch.cuda.is_available())
print(trigram[0], len(trigram))
# exit()


class n_gram(nn.Module):
    def __init__(self, vocab_size, context_size, n_dim):
        super(n_gram, self).__init__()
 
        self.embed = nn.Embedding(vocab_size, n_dim)   # (vocab_size,n_dim)
        self.classify = nn.Sequential(
            nn.Linear(context_size * n_dim, 128),   
            nn.ReLU(True),
            nn.Linear(128, vocab_size)
        )
 
    def forward(self, x):
        voc_embed = self.embed(x)  # 得到词嵌入  context_size*n_dim
        voc_embed = voc_embed.view(1, -1)  # 将两个词向量拼在一起  1*(context_size*n_dim)
        out = self.classify(voc_embed)   # 1*vocab_size
        return out
 
 
# 建立每个词与数字的编码，据此构建词嵌入
vocb = set(test_sentence)  # 使用 set 将重复的元素去掉
word_to_idx = {word: i for i, word in enumerate(vocb)}
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}
 
model = n_gram(len(word_to_idx), CONTEXT_SIZE, EMBEDDING_DIM)
criterion = nn.CrossEntropyLoss()

if torch.cuda.is_available():
	model = model.cuda()
if torch.cuda.is_available():
    criterion = criterion.cuda()

optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-5)
 
 
# for epoch in range(1000):
#     train_loss = 0
#     for word, label in trigram:
#         word = torch.LongTensor([word_to_idx[i] for i in word])  # 将两个词作为输入
#         label = torch.LongTensor([word_to_idx[label]])
#         if torch.cuda.is_available():
#             word = word.cuda()
#             label = label.cuda()
#         # 前向传播
#         out = model(word)
#         loss = criterion(out, label)
#         train_loss += loss.item()
#         # 反向传播
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     if (epoch + 1) % 20 == 0:
#         print('epoch: {}, Loss: {:.6f}'.format(epoch + 1, train_loss / len(trigram)))
#         torch.save(model,os.path.dirname(os.path.realpath(__file__))+'/tmp/st-b8-samsum-ngram_'+str(CONTEXT_SIZE)+'_e'+str(epoch+1)+'.pt')


# torch.save(model,os.path.dirname(os.path.realpath(__file__))+'/st-b8-samsum-ngram_'+str(CONTEXT_SIZE)+'.pt')
# print("Saved")

model = model.eval()
model = torch.load(os.path.dirname(os.path.realpath(__file__))+'/st-b8-samsum-ngram_'+str(CONTEXT_SIZE)+'.pt')
if torch.cuda.is_available():
	model = model.cuda()
        
word, label = trigram[15]
print('\ninput:{}'.format(word))
print('label:{}'.format(label))
word = torch.LongTensor([word_to_idx[i] for i in word])
if torch.cuda.is_available():
    word = word.cuda()
out = model(word)
pred_label_idx = out.max(1)[1].item()  # 第一行的最大值的下标
predict_word = idx_to_word[pred_label_idx]  # 得到对应的单词
print('real word is {}, predicted word is {}'.format(label, predict_word))
 