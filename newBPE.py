from collections import Counter, defaultdict
from transformers import AutoTokenizer
import re
class Byte_PairEncoding:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.bpe_vocab = {'<UNK>': 0, '<PAD>': 1}
    def train_bpe(self, corpus):
        print("Starting BPE training...")
        vocab = self.get_vocab(corpus)
        #print(f"Initial vocabulary size: {len(self.bpe_vocab)}")
        
        num_merges = 0
        while len(self.bpe_vocab) < self.vocab_size:
            # Get pair frequencies
            pairs = self.get_pair_freqs(vocab)
            if not pairs:
                print("No more pairs to merge")
                break
                
            # Find the most frequent pair
            most_frequent = max(pairs.items(), key=lambda x: x[1])[0]
            freq = pairs[most_frequent]
            print(f"\nMost frequent pair: {most_frequent} (frequency: {freq})")
            
            # Merge the pair in vocabulary
            new_vocab = {}
            merged = False
            
            for word, count in vocab.items():
                if not isinstance(word, tuple):
                    word = tuple(word)
                
                chars = list(word)
                i = 0
                new_chars = []
                
                while i < len(chars):
                    if i + 1 < len(chars) and chars[i] == most_frequent[0] and chars[i + 1] == most_frequent[1]:
                        new_chars.append(most_frequent[0] + most_frequent[1])
                        i += 2
                        merged = True
                    else:
                        new_chars.append(chars[i])
                        i += 1
                
                new_vocab[tuple(new_chars)] = count
            
            if not merged:
                print(f"No occurrences of pair {most_frequent} found to merge")
                break
                
            vocab = new_vocab
            
            # Add the merged token to vocabulary
            merged_token = ''.join(most_frequent)
            if merged_token not in self.bpe_vocab:
                idx = len(self.bpe_vocab)
                self.bpe_vocab[merged_token] = idx
                self.inverse_vocab[idx] = merged_token
                self.merges[most_frequent] = merged_token
                num_merges += 1
                print(f"Added new token: {merged_token} (index: {idx})")
                
            if num_merges % 100 == 0:
                print(f"Progress: {len(self.bpe_vocab)}/{self.vocab_size} tokens")
                
        print(f"\nTraining completed:")
        print(f"Final vocabulary size: {len(self.bpe_vocab)}")
        print(f"Number of merges performed: {num_merges}")
        return self.bpe_vocab
        
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
        
    def get_vocab(self, corpus):
        """
        Create initial vocabulary from corpus.
        """
        vocab = defaultdict(int)
        for text in corpus:
            # Split into words and handle each word separately
            for word in text.split():
                # Convert word into tuple of characters
                chars = tuple(list(word))
                vocab[chars] += 1
        
        print(f"Created initial vocabulary with {len(vocab)} entries")
        return dict(vocab)
        
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
    
    def encode(self, text):
        pre_tokenize_result = self.tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        splits_text = [[l for l in word] for word in pre_tokenized_text]

        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits_text):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
                splits_text[idx] = split
        result = sum(splits_text, [])
        token_indices = [self.bpe_vocab.get(subword, self.unk_token) for subword in result]
        return token_indices

    # def decode(self, tokens):
    #     decoded_words = []
    #     current_word = []
        
    #     for token in tokens:
    #         if token == self.pad_token:
    #             continue
            
    #         if token in self.inverse_vocab:
    #             word = self.inverse_vocab[token]
    #             if word == '<UNK>':
    #                 current_word.append('?')
    #             else:
    #                 current_word.append(word)
    #         else:
    #             current_word.append(f'<UNK-{token}>')
            
    #         # Check if we need to insert a space (optional)
    #         if len(current_word) > 1 and not current_word[-1].endswith('@'):
    #             decoded_words.append(''.join(current_word))
    #             current_word = []

    #     if current_word:
    #         decoded_words.append(''.join(current_word))
            
    #     return ' '.join(decoded_words)
    def decode(self, tokens):
        """
        Decode the tokenized text back into the original form using the inverse vocabulary.
        """
        decoded_words = []
        current_word = []

        for token in tokens:
            if token == self.pad_token:
                continue

            if token in self.inverse_vocab:
                word = self.inverse_vocab[token]
                current_word.append(word)
            else:
                current_word.append(f'<UNK-{token}>')

            # Check if it's time to finalize the word
            if len(current_word) > 1 and not current_word[-1].endswith('@'):
                decoded_words.append(''.join(current_word))
                current_word = []

        if current_word:
            decoded_words.append(''.join(current_word))

        return ' '.join(decoded_words)