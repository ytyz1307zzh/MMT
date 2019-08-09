'''
 @Date  : 8/22/2018
 @Author: Shuming Ma
 @mail  : shumingma@pku.edu.cn
 @homepage: shumingma.com
'''

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as Optim
from tensorboardX import SummaryWriter
import json
import argparse
import time
import os
import random
import pickle
from PIL import Image
import numpy as np
from modules import *


parser = argparse.ArgumentParser(description='train.py')
parser.add_argument('-n_emb', type=int, default=512, help="Embedding size")
parser.add_argument('-n_hidden', type=int, default=512, help="Hidden size")
parser.add_argument('-d_ff', type=int, default=2048, help="Hidden size of Feedforward")
parser.add_argument('-n_head', type=int, default=8, help="Number of head")
parser.add_argument('-n_block', type=int, default=6, help="Number of block")
parser.add_argument('-batch_size', type=int, default=64, help="Batch size")
parser.add_argument('-epoch', type=int, default=50, help="Number of epoch")
parser.add_argument('-impatience', type=int, default=10, help='number of evaluation rounds for early stopping')
parser.add_argument('-report', type=int, default=1000, help="Number of report interval")
parser.add_argument('-lr', type=float, default=3e-4, help="Learning rate")
parser.add_argument('-dropout', type=float, default=0.1, help="Dropout rate")
parser.add_argument('-restore', type=str, default='', help="Restoring model path")
parser.add_argument('-mode', type=str, default='train', help="Train or test")
parser.add_argument('-dir', type=str, default='ckpt', help="Checkpoint directory")
parser.add_argument('-max_len', type=int, default=30, help="Limited length for text")
parser.add_argument('-n_img', type=int, default=5, help="Number of input images")
parser.add_argument('-n_com', type=int, default=5, help="Number of input comments")
parser.add_argument('-output', default='prediction.json', help='Output json file for generation')
parser.add_argument('-src_lang', type=str, required=True, choices=['en', 'fr', 'de'], help='Source language')
parser.add_argument('-tgt_lang', type=str, required=True, choices=['en', 'fr', 'de'], help='Target language')
parser.add_argument('-reg_weight', type=float, default=2.0, help='Regularization Weight')

opt = parser.parse_args()
assert opt.src_lang != opt.tgt_lang
assert opt.src_lang == 'en' or opt.tgt_lang == 'en'

data_path = 'data/'
train_path = data_path + 'train_{}2{}.json'.format(opt.src_lang, opt.tgt_lang)
dev_path = data_path + 'val_{}2{}.json'.format(opt.src_lang, opt.tgt_lang)
src_vocab_path = data_path + '{}_dict.json'.format(opt.src_lang)
tgt_vocab_path = data_path + '{}_dict.json'.format(opt.tgt_lang)
train_img_path, dev_img_path = data_path + 'train_res34.pkl', data_path + 'val_res34.pkl'

src_vocabs = json.load(open(src_vocab_path, 'r', encoding='utf8'))['word2id']
tgt_vocabs = json.load(open(tgt_vocab_path, 'r', encoding='utf8'))['word2id']
src_rev_vocabs = json.load(open(src_vocab_path, 'r', encoding='utf8'))['id2word']
tgt_rev_vocabs = json.load(open(tgt_vocab_path, 'r', encoding='utf8'))['id2word']
opt.src_vocab_size = len(src_vocabs)
opt.tgt_vocab_size = len(tgt_vocabs)

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

if not os.path.exists(opt.dir):
    os.mkdir(opt.dir)

class Model(nn.Module):

    def __init__(self, n_emb, n_hidden, src_vocab_size, tgt_vocab_size, dropout, d_ff, n_head, n_block):
        super(Model, self).__init__()
        self.n_emb = n_emb
        self.n_hidden = n_hidden
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dropout = dropout
        self.src_embedding = nn.Sequential(Embeddings(n_hidden, src_vocab_size), PositionalEncoding(n_hidden, dropout))
        self.tgt_embedding = nn.Sequential(Embeddings(n_hidden, tgt_vocab_size), PositionalEncoding(n_hidden, dropout))
        self.video_encoder = VideoEncoder(n_hidden, d_ff, n_head, dropout, n_block)
        self.text_encoder = TextEncoder(n_hidden, d_ff, n_head, dropout, n_block)
        self.comment_decoder = CommentDecoder(n_hidden, d_ff, n_head, dropout, n_block)
        self.output_layer = nn.Linear(self.n_hidden, self.tgt_vocab_size)
        self.criterion = nn.CrossEntropyLoss(reduce=False, size_average=False, ignore_index=0)
        self.co_attn = CoAttention(n_hidden)

    def encode_img(self, X):
        out = self.video_encoder(X)
        return out

    def encode_text(self, X, m):
        embs = self.src_embedding(X)
        out = self.text_encoder(embs, m)
        return out

    def decode(self, x, m1, m2, mask):
        embs = self.tgt_embedding(x)
        H_v_hat, H_x_hat = self.co_attn(m1, m2)
        out = self.comment_decoder(embs, m1, m2, H_v_hat, H_x_hat, mask)
        out = self.output_layer(out)
        return out

    def forward(self, X, Y, T):
        out_img = self.encode_img(X)
        out_text = self.encode_text(T, out_img)
        mask = Variable(subsequent_mask(Y.size(0), Y.size(1)-1), requires_grad=False).cuda()
        outs = self.decode(Y[:,:-1], out_img, out_text, mask)

        Y = Y.t()
        outs = outs.transpose(0, 1)

        loss = self.criterion(outs.contiguous().view(-1, self.tgt_vocab_size),
                              Y[1:].contiguous().view(-1))

        return torch.mean(loss)


    def generate(self, X, T):
        out_img = self.encode_img(X)
        out_text = self.encode_text(T, out_img)

        ys = torch.ones(X.size(0), 1).long()
        with torch.no_grad():
            ys = Variable(ys).cuda()
        for i in range(opt.max_len):
            out = self.decode(ys, out_img, out_text,
                              Variable(subsequent_mask(ys.size(0), ys.size(1))).cuda())
            prob = out[:, -1]
            _, next_word = torch.max(prob, dim=-1, keepdim=True)
            next_word = next_word.data
            ys = torch.cat([ys, next_word], dim=-1)

        return ys[:, 1:]

class DataSet(torch.utils.data.Dataset):

    def __init__(self, data_path, src_vocabs, tgt_vocabs, img_path, is_train=True):
        print("starting load...")
        start_time = time.time()
        print('load data from file: ', data_path)
        print('load images from file: ', img_path)
        self.datas = json.load(open(data_path, 'r', encoding='utf8')) # each piece is a dict like {"src": xxx, "tgt": xxx}
        self.imgs = torch.load(open(img_path, 'rb'))
        print("loading time:", time.time() - start_time)

        self.src_vocabs = src_vocabs
        self.tgt_vocabs = tgt_vocabs
        self.src_vocab_size = len(self.src_vocabs)
        self.tgt_vocab_size = len(self.tgt_vocabs)
        self.is_train = is_train

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index] #  a dict like {"src": xxx, "tgt": xxx}

        I = torch.cuda.FloatTensor(self.imgs[index]['features'])  # input image
        I = I.view(I.size(0), -1)
        X = torch.stack([I[:, i] for i in range(I.size(1))])
        T = DataSet.padding(data['src'], opt.max_len, 'src') # source sequence
        Y = DataSet.padding(data['tgt'], opt.max_len, 'tgt') # target sequence

        return X, Y, T

    @staticmethod
    # cut sentences that exceed the limit, turn words into numbers, pad sentences to max_len
    def padding(data, max_len, language):
        if language == 'src': # source language
            vocabs = src_vocabs
        elif language == 'tgt': # target language
            vocabs = tgt_vocabs

        data = data.split()
        if len(data) > max_len-2:
            data = data[:max_len-2]
        Y = list(map(lambda t: vocabs.get(t, 3), data))
        Y = [1] + Y + [2]
        length = len(Y)
        Y = torch.cat([torch.LongTensor(Y), torch.zeros(max_len - length).long()])
        return Y

    @staticmethod
    def transform_to_words(ids, language):
        if language == 'src': # source language
            rev_vocabs = src_rev_vocabs
        elif language == 'tgt': # target language
            rev_vocabs = tgt_rev_vocabs

        words = []
        for id in ids:
            if id == 2:
                break
            words.append(rev_vocabs[str(id.item())])
        return " ".join(words)


def get_dataloader(dataset, batch_size, is_train=True):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_train)

def save_model(path, model, rev_model):
    model_state_dict = model.state_dict()
    rev_model_state_dict = rev_model.state_dict()
    torch.save({'model': model_state_dict,
                'rev_model': rev_model_state_dict} , path)


def train():
    train_set = DataSet(train_path, src_vocabs, tgt_vocabs, train_img_path, is_train=True)
    dev_set = DataSet(dev_path, src_vocabs, tgt_vocabs, dev_img_path, is_train=False)
    train_batch = get_dataloader(train_set, opt.batch_size, is_train=True)
    model = Model(n_emb=opt.n_emb, n_hidden=opt.n_hidden, src_vocab_size=opt.src_vocab_size, tgt_vocab_size=opt.tgt_vocab_size,
                  dropout=opt.dropout, d_ff=opt.d_ff, n_head=opt.n_head, n_block=opt.n_block)
    rev_model = Model(n_emb=opt.n_emb, n_hidden=opt.n_hidden, src_vocab_size=opt.tgt_vocab_size, tgt_vocab_size=opt.src_vocab_size,
                  dropout=opt.dropout, d_ff=opt.d_ff, n_head=opt.n_head, n_block=opt.n_block)
    if opt.restore != '':
        saved_data = torch.load(opt.restore)
        model_dict = saved_data['model']
        rev_model_dict = saved_data['rev_model']
        model.load_state_dict(model_dict)
        rev_model.load_state_dict(rev_model_dict)

    model.cuda()
    rev_model.cuda()
    optim = Optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=opt.lr)
    rev_optim = Optim.Adam(filter(lambda p: p.requires_grad,rev_model.parameters()), lr=opt.lr)
    best_score = -1000000
    impatience = 0
    #writer = SummaryWriter()

    for i in range(opt.epoch):
        model.train()
        rev_model.train()
        report_loss, start_time, n_samples = 0, time.time(), 0
        report_rev_loss, report_joint_loss = 0, 0
        count, total = 0, len(train_set) // opt.batch_size + 1
        for batch in train_batch:
            X, Y, T = batch # X:images, Y: target, T: source
            X = Variable(X).cuda()
            Y = Variable(Y).cuda()
            T = Variable(T).cuda()

            loss = model(X, Y, T)
            rev_loss = rev_model(X, T, Y) # exchange the source and the target
            attn = model.comment_decoder.layers[0].text_attn.attn
            rev_attn = rev_model.comment_decoder.layers[0].text_attn.attn
            attn = torch.mean(attn, dim=1)[:, :, :-1] # average on 8 heads, omit <EOS> on the last position
            rev_attn = torch.mean(rev_attn, dim=1)[:, :, :-1]
            #print('attn: ', attn.size())
            #print('attn: ', attn[0][:5])
            #print('rev_attn: ', rev_attn.size())
            #print('rev_attn: ', rev_attn[0][:5])
            attn_product = torch.mul(attn, rev_attn.transpose(1, 2))
            #print('attn_product: ', attn_product[0][:5])
            attn_sum = torch.sum(attn_product, dim=[1, 2]) # calculate the sum of attention weights for each instance
            #print('attn_sum: ', attn_sum)
            attn_log = torch.log2(attn_sum)
            #print('attn_log: ', attn_log)
            joint_loss = torch.mean(opt.reg_weight * attn_log)
            if count % 10 == 0:
                print('loss: ', loss.item(), end=' ')
                print('rev_loss: ', rev_loss.item(), end=' ')
                print('joint_loss: ', joint_loss.item())

            #writer.add_scalars('tensorboard/loss', {'loss': loss.item(), 'rev_loss': rev_loss.item(), 'joint_loss': joint_loss.item()}, count)

            optim.zero_grad()
            loss_forward = loss - joint_loss
            loss_backward = rev_loss - joint_loss
            loss_forward.backward(retain_graph=True)
            optim.step()
            rev_optim.zero_grad()
            loss_backward.backward()
            rev_optim.step()

            report_loss += loss.item()
            report_rev_loss += rev_loss.item()
            report_joint_loss += joint_loss.item()
            n_samples += len(X.data)
            count += 1
            if count % opt.report == 0 or count == total:
                print('%d/%d, epoch: %d, loss: %.3f, rev_loss: %.3f, joint_loss: %.3f, time: %.2f'
                      % (count, total, i+1, report_loss / n_samples, report_rev_loss / n_samples, report_joint_loss / n_samples, time.time() - start_time))
                model.eval()
                rev_model.eval()
                score = eval(dev_set, model, rev_model)
                model.train()
                rev_model.train()

                if score > best_score:
                    best_score = score
                    impatience = 0
                    print('New best score!')
                    save_model(os.path.join(opt.dir, 'best_checkpoint_{:.3f}.pt'.format(-score)), model, rev_model)
                else:
                    impatience += 1
                    print('Impatience: ', impatience, 'best score: ', best_score)
                    save_model(os.path.join(opt.dir, 'impatience_{:.3f}.pt'.format(-score)), model, rev_model)
                    if impatience > opt.impatience:
                        print('Early stopping!')
                        quit()

                report_loss, start_time, n_samples = 0, time.time(), 0
                report_rev_loss, report_joint_loss = 0, 0
        #save_model(os.path.join(opt.dir, 'checkpoint_{}.pt'.format(i+1)), model)
    #writer.close()

    return model

def eval(dev_set, model, rev_model):
    print("starting evaluating...")
    start_time = time.time()
    model.eval()
    rev_model.eval()
    dev_batch = get_dataloader(dev_set, opt.batch_size, is_train=False)

    loss = 0
    rev_loss = 0
    for batch in dev_batch:
        X, Y, T = batch
        with torch.no_grad():
            X = Variable(X).cuda()
            Y = Variable(Y).cuda()
            T = Variable(T).cuda()
        loss += model(X, Y, T).item()
        rev_loss += rev_model(X, T, Y).item()

    loss = (loss * opt.batch_size) / 64
    rev_loss = (rev_loss * opt.batch_size) / 64

    print('loss: ', loss)
    print('rev_loss: ', rev_loss)
    print("evaluating time:", time.time() - start_time)

    return -loss

def test(test_set, model, rev_model):
    model.eval()
    rev_model.eval()
    test_batch = get_dataloader(test_set, opt.batch_size, is_train=False)
    assert opt.output.endswith('.json'), 'Output file should be a json file'
    outputs = []
    cnt = 0 # counter for testing process

    for batch in test_batch:
        X, Y, T = batch
        cnt += X.size()[0]
        with torch.no_grad():
            X = Variable(X).cuda()
            Y = Variable(Y).cuda()
            T = Variable(T).cuda()
        predictions = model.generate(X, T).data
        rev_predictions = rev_model.generate(X, Y).data

        assert X.size()[0] == predictions.size()[0] and X.size()[0] == T.size()[0]
        for i in range(X.size()[0]):
            out_dict = {'source': DataSet.transform_to_words(T[i].cpu(), 'src'),
                        'target': DataSet.transform_to_words(Y[i].cpu(), 'tgt'),
                        'prediction': DataSet.transform_to_words(predictions[i].cpu(), 'tgt'),
                        'rev_prediction': DataSet.transform_to_words(rev_predictions[i].cpu(), 'src')}
            outputs.append(out_dict)

        print(cnt)
    json.dump(outputs, open(opt.output, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print('All data finished.')



if __name__ == '__main__':
    print(opt)
    if opt.mode == 'train':
        train()
    elif opt.mode == 'test':
        test_path = data_path + 'test_{}2{}.json'.format(opt.src_lang, opt.tgt_lang)
        test_img_path = data_path + 'test_res34.pkl'
        test_set = DataSet(test_path, src_vocabs, tgt_vocabs, test_img_path, is_train=False)
        model = Model(n_emb=opt.n_emb, n_hidden=opt.n_hidden, src_vocab_size=opt.src_vocab_size, tgt_vocab_size=opt.tgt_vocab_size,
                  dropout=opt.dropout, d_ff=opt.d_ff, n_head=opt.n_head, n_block=opt.n_block)
        rev_model = Model(n_emb=opt.n_emb, n_hidden=opt.n_hidden, src_vocab_size=opt.tgt_vocab_size,tgt_vocab_size=opt.src_vocab_size,
                          dropout=opt.dropout, d_ff=opt.d_ff, n_head=opt.n_head, n_block=opt.n_block)
        saved_data = torch.load(opt.restore)
        model_dict = saved_data['model']
        rev_model_dict = saved_data['rev_model']
        model.load_state_dict(model_dict)
        rev_model.load_state_dict(rev_model_dict)
        model.cuda()
        rev_model.cuda()
        test(test_set, model, rev_model)

