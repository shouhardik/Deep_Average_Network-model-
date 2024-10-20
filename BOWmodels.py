# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, WordEmbeddings
from BPEmodel import BytePairEncoding as bpe
from torch.utils.data import Dataset


# Dataset class for handling sentiment analysis data
class SentimentDatasetBOW(Dataset):
    def __init__(self, infile, vectorizer=None, train=True):
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)
        
        # Extract sentences and labels from the examples
        self.sentences = [" ".join(ex.words) for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]
        
        # Vectorize the sentences using CountVectorizer
        if train or vectorizer is None:
            self.vectorizer = CountVectorizer(max_features=512)
            self.embeddings = self.vectorizer.fit_transform(self.sentences).toarray()
        else:
            self.vectorizer = vectorizer
            self.embeddings = self.vectorizer.transform(self.sentences).toarray()
        
        # Convert embeddings and labels to PyTorch tensors
        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return the feature vector and label for the given index
        return self.embeddings[idx], self.labels[idx]


# Two-layer fully connected neural network
class NN2BOW(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x

    
# Three-layer fully connected neural network
class NN3BOW(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.log_softmax(x)


class SentimentDatasetDAN(Dataset):
    def __init__(self, file_path, word_embeddings: WordEmbeddings, max_len=50):
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(file_path)
        
        # Extract sentences and labels from the examples
        self.sentences = [ex.words for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]

        print(self.sentences[:1])
        print(self.labels[:1])

        self.word_index = [[word_embeddings.get_embedding(word) for word in sentence] for sentence in self.sentences]
        for i, sentence in enumerate(self.word_index):
            if len(sentence) > max_len:
                self.word_index[i] = sentence[:max_len]
            else:
                self.word_index[i] += [word_embeddings.get_embedding('<UNK>')] * (max_len - len(sentence))
        # print(self.word_index[:3])

        self.sentences = torch.tensor(self.word_index)
        self.labels = torch.tensor(self.labels)

        # self.sentences, self.labels = self.load_data(file_path)
        # self.word_to_index = word_to_index
        # self.max_len = max_len



    def load_data(self, file_path):
        sentences = []
        labels = []
        with open(file_path, 'r') as f:
            for line in f:
                label, sentence = line.strip().split('\t')
                sentences.append(sentence.split()) 
                labels.append(int(label))
        return sentences, labels
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        return sentence, label



class SentimentDatasetBPE(Dataset):
    def __init__(self, file_path, bpe, bpe_vocab, max_len=None):
        self.bpe_vocab = bpe_vocab
        self.bpe = bpe
        self.max_len = max_len
        self.sentences, self.labels = self._load_data(file_path)  # Uses this method
        self.unk_token = 0  # Using 0 as UNK token index
        self.pad_token = 1
        

    # This is the method you already have:
    def _load_data(self, infile):
        sentences = []
        labels = []
        with open(infile, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.split('\t', 1)
                label = int(parts[0])
                sentence = parts[1].strip()
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
        #self.sentences = [" ".join()]
        print("My sentences: "+str(sentences[:5]))
        print("My labels: "+str(labels[:5]))
        return sentences, labels

    def tokenize_sentence(self, sentence):
        """Apply BPE tokenization to a sentence"""
        return self.bpe.encode(sentence)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        
        encoded_sentence = self.tokenize_sentence(sentence)
        
        # Pad or truncate
        if self.max_len:
            if len(encoded_sentence) > self.max_len:
                encoded_sentence = encoded_sentence[:self.max_len]
            else:
                encoded_sentence += [self.pad_token] * (self.max_len - len(encoded_sentence))
        
        return torch.tensor(encoded_sentence, dtype=torch.long), torch.tensor(label)
    
    def decodeSentence(self, encoded_sentence):
        """
        Decode an encoded sentence back to text.
        """
        return self.bpe.decode(encoded_sentence)

    def __len__(self):
        return len(self.sentences)