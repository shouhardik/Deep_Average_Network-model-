import re
from collections import Counter, defaultdict
from transformers import AutoTokenizer
import re

class BytePairEncoding:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.bpe_vocab = {'<UNK>': 0, '<PAD>': 1}
        self.inverse_vocab = {0: '<UNK>', 1: '<PAD>'}
        self.unk_token = 0
        self.pad_token = 1
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.word_freqs = defaultdict(int)
        self.splits = {}
        self.merges = {}

    def get_vocab(self, corpus):
        words = corpus.split()
        return Counter([''.join(list(word)) for word in words])

    def get_pair_freqs(self, vocab):
        """
        Count frequency of all symbol pairs in vocabulary.
        """
        pairs = defaultdict(int)
        
        for word, freq in vocab.items():
            if not isinstance(word, tuple):
                word = tuple(word)
            
            chars = list(word)
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                pairs[pair] += freq
        
        return dict(pairs)


    def merge_pair(self, pair, vocab):
        """
        Merge a pair of tokens in the vocabulary.
        """
        merged_token = ''.join(pair)
        new_vocab = {}
        
        for word, freq in vocab.items():
            if not isinstance(word, tuple):
                word = tuple(word)
            
            chars = list(word)
            i = 0
            new_chars = []
            
            while i < len(chars):
                if i + 1 < len(chars) and chars[i] == pair[0] and chars[i + 1] == pair[1]:
                    new_chars.append(merged_token)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            
            new_vocab[tuple(new_chars)] = freq
        
        return new_vocab    

    # def train_bpe(self, corpus):
    #     self.vocab = self.get_vocab(corpus)
    #     print(f"Initial vocab size: {len(self.vocab)}")
        
    #     # Initialize character vocabulary
    #     unique_chars = set(''.join(self.vocab.keys()))
    #     self.bpe_vocab = {char: idx for idx, char in enumerate(sorted(unique_chars))}
        
    #     num_merges = 0
    #     max_iterations = 10000  # Add a maximum number of iterations
    #     iteration = 0
    #     #num_merges < 1000:
    #     while len(self.bpe_vocab) < self.vocab_size and num_merges < 1000:
    #         iteration += 1
    #         # Get pair frequencies
    #         pairs = self.get_pair_freqs(self.vocab)
    #         if not pairs:
    #             print("No more pairs to merge")
    #             break
            
    #         # Find the most frequent pair
    #         most_frequent = max(pairs.items(), key=lambda x: x[1])[0]
    #         freq = pairs[most_frequent]
    #         #print(f"\nIteration {iteration}: Most frequent pair: {most_frequent} (frequency: {freq})")
            
    #         # Merge the pair in vocabulary
    #         new_vocab = {}
    #         merged = False
            
    #         for word, count in self.vocab.items():
    #             if not isinstance(word, tuple):
    #                 word = tuple(word)
                
    #             chars = list(word)
    #             i = 0
    #             new_chars = []
                
    #             while i < len(chars):
    #                 if i + 1 < len(chars) and chars[i] == most_frequent[0] and chars[i + 1] == most_frequent[1]:
    #                     new_chars.append(most_frequent[0] + most_frequent[1])
    #                     i += 2
    #                     merged = True
    #                 else:
    #                     new_chars.append(chars[i])
    #                     i += 1
                
    #             new_vocab[tuple(new_chars)] = count
            
    #         if not merged:
    #             print(f"No occurrences of pair {most_frequent} found to merge")
    #             break
            
    #         self.vocab = new_vocab  # Update the main vocabulary
            
    #         # Add the merged token to vocabulary
    #         merged_token = ''.join(most_frequent)
    #         if merged_token not in self.bpe_vocab:
    #             idx = len(self.bpe_vocab)
    #             self.bpe_vocab[merged_token] = idx
    #             self.inverse_vocab[idx] = merged_token
    #             self.merges[most_frequent] = merged_token
    #             num_merges += 1
    #             #print(f"Added new token: {merged_token} (index: {idx})")
            
    #         if num_merges % 100 == 0:
    #             print(f"Progress: {len(self.bpe_vocab)}/{self.vocab_size} tokens")
        
    #     print(f"\nTraining completed:")
    #     print(f"Final vocabulary size: {len(self.bpe_vocab)}")
    #     print(f"Number of merges performed: {num_merges}")
    #     return self.bpe_vocab

    # def train_bpe(self, corpus):
    #     # Get initial vocabulary
    #     words = corpus.split()
    #     word_freqs = defaultdict(int)
    #     for word in words:
    #         chars = list(word)
    #         word_freqs[' '.join(chars)] += 1
        
    #     # Initialize vocabulary with characters
    #     for word in word_freqs.keys():
    #         for char in word.split():
    #             if char not in self.bpe_vocab:
    #                 self.bpe_vocab[char] = len(self.bpe_vocab)
    #                 self.inverse_vocab[len(self.inverse_vocab)] = char
        
    #     num_merges = 0
    #     while len(self.bpe_vocab) < self.vocab_size and num_merges < 2000:
    #         # Get pair frequencies
    #         pair_freqs = defaultdict(int)
    #         for word, freq in word_freqs.items():
                
    #             symbols = word.split()
    #             for i in range(len(symbols) - 1):
    #                 pair = (symbols[i], symbols[i + 1])
    #                 pair_freqs[pair] += freq
            
    #         if not pair_freqs:
    #             break
            
            
                
    #         # Find most frequent pair
    #         best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
    #         #best_pair = max(pair_freqs.items(), key=lambda x: x[1])
    #         #actual_best_pair = best_pair[0]

    #         best_freq = pair_freqs[best_pair]
    #         #print(f"Merging pair: {best_pair} (frequency: {best_freq})")

    #         # Merge pair in all words
    #         new_word_freqs = defaultdict(int)
    #         bigram = ' '.join(best_pair)
    #         replacement = ''.join(best_pair)
            
    #         for word, freq in word_freqs.items():
    #             new_word = word.replace(bigram, replacement)
    #             new_word_freqs[new_word] = freq
            
    #         word_freqs = new_word_freqs
            
    #         # Add merged token to vocabulary
    #         if replacement not in self.bpe_vocab:
    #             self.bpe_vocab[replacement] = len(self.bpe_vocab)
    #             self.inverse_vocab[len(self.inverse_vocab)] = replacement
    #             self.merges[best_pair] = replacement
    #             num_merges += 1
            
    #         if num_merges % 100 == 0:
    #             print(f"Progress: {len(self.bpe_vocab)}/{self.vocab_size} tokens")
        
    #     return self.bpe_vocab

    def train_bpe(self, corpus):
        words = corpus.split()
        word_freqs = defaultdict(int)
        for word in words:
            chars = list(word)
            word_freqs[' '.join(chars)] += 1
        for word in word_freqs.keys():
            for char in word.split():
                if char not in self.bpe_vocab:
                    self.bpe_vocab[char] = len(self.bpe_vocab)
                    self.inverse_vocab[len(self.inverse_vocab)] = char
        num_merges = 0
        while len(self.bpe_vocab) < self.vocab_size and num_merges < 1000:
            pair_freqs = self.get_pair_freqs(word_freqs)
            if not pair_freqs:
                break
            best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
            word_freqs = self.merge_pair(best_pair, word_freqs)
            merged_token = ''.join(best_pair)
            if merged_token not in self.bpe_vocab:
                self.bpe_vocab[merged_token] = len(self.bpe_vocab)
                self.inverse_vocab[len(self.inverse_vocab)] = merged_token
                num_merges += 1
            if num_merges % 100 == 0:
                print(f"Progress: {len(self.bpe_vocab)}/{self.vocab_size} tokens")
        return self.bpe_vocab



    # def encode(self, text):
    #     pre_tokenize_result = self.tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    #     pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    #     splits_text = [[l for l in word] for word in pre_tokenized_text]

    #     for pair, merge in self.merges.items():
    #         for idx, split in enumerate(splits_text):
    #             i = 0
    #             while i < len(split) - 1:
    #                 if split[i] == pair[0] and split[i + 1] == pair[1]:
    #                     split = split[:i] + [merge] + split[i + 2 :]
    #                 else:
    #                     i += 1
    #             splits_text[idx] = split
    #     result = sum(splits_text, [])
    #     token_indices = [self.bpe_vocab.get(subword, self.unk_token) for subword in result]
    #     return token_indices

   
    # def decode(self, tokens):
    #     """
    #     Decode the tokenized text back into the original form using the inverse vocabulary.
    #     """
    #     decoded_words = []
    #     current_word = []

    #     for token in tokens:
    #         if token == self.pad_token:
    #             continue

    #         if token in self.inverse_vocab:
    #             word = self.inverse_vocab[token]
    #             current_word.append(word)
    #         else:
    #             current_word.append(f'<UNK-{token}>')

    #         # Check if it's time to finalize the word
    #         if len(current_word) > 1 and not current_word[-1].endswith('@'):
    #             decoded_words.append(''.join(current_word))
    #             current_word = []

    #     if current_word:
    #         decoded_words.append(''.join(current_word))

    #     return ' '.join(decoded_words)
    def encode(self, text):
        # Pre-tokenize
        words = self.tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        encoded = []
        
        for word, _ in words:
            chars = list(word)
            while len(chars) > 1:
                pairs = [(chars[i], chars[i+1]) for i in range(len(chars)-1)]
                replacements = [(i, pair) for i, pair in enumerate(pairs) if pair in self.merges]
                
                if not replacements:
                    break
                    
                i, pair = min(replacements)
                #i, pair = max(replacements, key=lambda x: self.merges[x[1]])
                chars = chars[:i] + [self.merges[pair]] + chars[i+2:]
            
            for char in chars:
                encoded.append(self.bpe_vocab.get(char, self.unk_token))
                
        return encoded

    def decode(self, tokens):
        decoded = []
        for token in tokens:
            if token == self.pad_token:
                continue
            decoded.append(self.inverse_vocab.get(token, f'<UNK-{token}>'))
        return ''.join(decoded)

