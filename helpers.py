import pickle, argparse, os, sys
from sklearn.metrics import accuracy_score
import numpy as np
import random
import torch
import torch.nn as nn
import torch.functional as F
from collections import defaultdict
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from helpers import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data_with_tags(path):
    '''
    reads the raw files from the path 
    replaces line endings with specific token and tag
    '''
    with open(path, 'r') as f:
        data = f.read()
    
    # because we want to keep track where the sentence starts and ends
    data = data.replace('\n', " --n-- --s-- ")
    
    # split by evertyhing
    # potentional source of bugs
    splitted = data.split()

    pairs = list()
    for i in range(0, len(splitted) , 2):
        # we use lists instead of tuples because we want to change it to rare words later
        pairs.append([splitted[i], splitted[i+1]])
    
    return pairs

def build_vocab_replace_rare(train_pairs):
    '''
    :param train_pairs: pairs in form (token, pos)
    :return:
    '''
    voc = defaultdict(int)

    # first iterate over the words and count
    for token, pos_tag in train_pairs:
        voc[token] += 1

    # find the rare words
    oov_words = [k for k, v in voc.items() if v < 3]
    oov_words = set(oov_words)

    # manual check
    # 'Vinken' in oov_words
    
    # replace rare words with "UNKA" token
    # O(NP) where N is number of trainign example and T is number of out of vocab words
    for pair in train_pairs:
        if pair[0] in oov_words:
            pair[0] = 'UNKA'
    
    
    # Now rebuild the vocabulary to take UNKA into account
    voc = defaultdict(int)
    for token, pos_tag in train_pairs:
        voc[token] += 1
    oov_words = [k for k, v in voc.items() if v < 3]

    # no rare words anymore
    assert len(oov_words) == 0

    # make a list
    voc = [k for k, v in voc.items()]

    return voc, train_pairs

def get_glove(glove_fpath, embd_size, vocab_size, word_index):
    embeddings_index = {}
    with open(glove_fpath, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings_index[word] = vector
    
    print('Found {} word vectors in glove.'.format(len(embeddings_index)))
    embedding_matrix = np.zeros((vocab_size, embd_size))
    print('embed_matrix.shape', embedding_matrix.shape)
    found_ct = 0
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        # words not found in embedding index will be all-zeros.
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            found_ct += 1
    print('{} words are found in glove'.format(found_ct))

    return embedding_matrix

def make_validation_pairs(data_file='data/wsj19-21.testing', label_file='data/wsj19-21.truth'):
    with open(data_file, 'r') as f:
        test = f.read()

    # end of line tag
    test = test.replace('\n', " --n-- ")

    # simply split because there are no tags
    test = test.split()

    ground_truth_pairs = load_data_with_tags(label_file)
    ground_truth = [i[1] for i in ground_truth_pairs]

    # make sentences instead of random chunks
    valid_sentences = list()
    sentence = list()
    for x, y in zip(test, ground_truth):
        if x == '--n--' and y == '--s--':
            sentence.append((x, y))
            valid_sentences.append(sentence)
            sentence = list()
        else:
            sentence.append((x, y))
            
    return valid_sentences

def get_training_data(training_file='data/wsj1-18.training'):
    # get data in pairs (token, pos)
    train_pairs = load_data_with_tags(training_file)

    # remove rare words and build vocab
    voc, train_pairs = build_vocab_replace_rare(train_pairs)
    
    # build indices
    word_2_idx = {tok:ix for ix, tok in enumerate(voc, 1)}
    word_2_idx['<PAD>'] = 0
    idx_2_word = {v: k for k, v in word_2_idx.items()}
    
    # make sentences instead of chunks
    sentences = list()
    sentence = list()
    for x, y in train_pairs:
        if x == '--n--' and y == '--s--':
            sentence.append((x, y))
            sentences.append(sentence)
            sentence = list()
        else:
            sentence.append((x, y))
    
    # build indices for tags as well
    unique_tags = np.unique([i[1] for i in train_pairs])
    target_size = len(unique_tags) + 1 # because we add pad

    # make the same dictionary for tag
    tag_mapper = dict()
    for i, tag in enumerate(unique_tags, 1):
        tag_mapper[tag] = i

    # this will be ignored in our target value
    tag_mapper['<PAD>'] = 0
    
    return sentences, voc, idx_2_word, word_2_idx, tag_mapper, target_size

class LSTMTagger(nn.Module):
    def __init__(self, weights_matrix, hidden_dim, tagset_size, trained=False):
        super(LSTMTagger, self).__init__()
        
        # embeddings 
        vocab_size, embedding_dim = weights_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.from_pretrained(torch.Tensor(weights_matrix), freeze=True)
        
        # LSTM 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, text):
        embedded = self.embedding(text)    
        lstm_out, _ = self.lstm(embedded)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = self.softmax(tag_space)
        return tag_scores
    

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean = 0, std = 0.1)
        
class WSJDataset(Dataset):
    """Custom WSJ Dataset class"""

    def __init__(self, sentences, vocab_mapper, tag_mapper):
        self.xs = [[vocab_mapper[t[0]] for t in s] for s in sentences]
        self.ys =[[tag_mapper[t[1]] for t in s] for s in sentences]
        
    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx): 
        return self.xs[idx], self.ys[idx]
    
def collate_examples(batch, truncate_len=512):
    
    """Batch preparation.
    1. Pad the sequences
    2. Transform the target.
    """
    
    transposed = list(zip(*batch))
    
    max_len = min(
        max((len(x) for x in transposed[0])),
        truncate_len
    )
    
    tokens = np.zeros((len(batch), max_len), dtype=np.int64)
    tags = np.zeros((len(batch), max_len), dtype=np.int64)
    
    # tokens
    for i, row in enumerate(transposed[0]):
        row = np.array(row[:truncate_len])
        tokens[i, :len(row)] = row
    token_tensor = torch.from_numpy(tokens)
    
    # tags
    # possible bug?
    for i, row in enumerate(transposed[1]):
        row = np.array(row[:truncate_len])
        tags[i, :len(row)] = row
    tags_tensor = torch.from_numpy(tags)
    
    return token_tensor, tags_tensor

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    non_pad_elements = (y != 0).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])

def train_epoch(model, iterator, optimizer, criterion, epoch):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch_idx, (text, tags) in enumerate(iterator):
        
        text = text.to(device)
        

        predictions = model(text)
        
        predictions = predictions.view(-1, predictions.shape[-1]).cpu()
        tags = tags.view(-1)
        
        loss = criterion(predictions, tags)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = categorical_accuracy(predictions, tags)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
        if not batch_idx % 100:
            print (f'Epoch: {epoch+1:03d}/{10:03d} | ' 
                   f'Batch {batch_idx:03d}/{len(iterator):03d} | '
                   f'Cost: {loss:.4f} |', 
                   f': Accuracy: {acc.item():.4f}') 
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def make_predictions(model, iterator):
    '''
    notice: I am using the labels here because I don't want to write separate class. 
    They are not used and just passed further
    '''
    model.eval()
    all_predicted = list()
    all_tags = list()
    
    with torch.no_grad():
    
        for batch_idx, (text, tags) in enumerate(iterator):

            text = text.to(device)
            predictions = model(text)

            predictions = predictions.argmax(-1)
            predictions = predictions.view(-1, predictions.shape[-1]).cpu().numpy()
            
            # pads should be excluded
            tags = tags.numpy().ravel()
            non_pad_elements = (tags != 0).nonzero()
    
            all_tags.append(tags[non_pad_elements])
            all_predicted.append(predictions.ravel()[non_pad_elements])
            
    return all_predicted, all_tags