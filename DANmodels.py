import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class DAN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_layer, dropout_rate):

        super().__init__()
        self.embedding = embedding_layer
        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(embedding_layer.embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Assuming binary classification (2 output classes)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        """
        :param x: Input tensor (batch_size x sequence_length)
        """
        # Convert word indices to embeddings and then take the mean across the sequence
        # embedded = self.embedding(x)  # Shape: (batch_size, sequence_length, embedding_dim)
        x = torch.mean(x, dim=1)  # Averaging embeddings (Shape: batch_size x embedding_dim)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))  # Hidden layer
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        # Log softmax for classification
        return self.log_softmax(x)


# class DAN_BPE(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, embedding_layer, output_dim):
#         super(DAN_BPE, self).__init__()
#         self.embedding = embedding_layer
#         print("DAN-BPE Initialization:")
#         print(f"Vocabulary Size: {vocab_size}")
#         print(f"Embedding Dimension: {embedding_dim}")
#         print(f"Output Dimension: {output_dim}")
#         self.output_dim = output_dim
#         self.fc = nn.Linear(embedding_dim, output_dim)

#     def forward(self, x):
#         if torch.max(x) >= self.embedding.num_embeddings:
#             print("Error: Input indices exceed vocabulary size")
#             print(f"Max index in batch: {torch.max(x)}")
#             print(f"Vocabulary size: {self.embedding.num_embeddings}")
#             raise ValueError("Input indices exceed vocabulary size")
#         x = x.long()
#         # print(x)
#         # Get word embeddings for input token ids
#         embedded = self.embedding(x)  # (batch_size, max_len, embedding_dim)
        
#         # Average the embeddings along the sequence dimension (axis 1)
#         averaged_embeds = torch.mean(embedded, dim=1)  # (batch_size, embedding_dim)

#         # Pass through fully connected layer for classification
#         logits = self.fc(averaged_embeds)  # (batch_size, output_dim)
        
#         return logits

class DAN_BPE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_layer, output_dim, hidden_dim=128):
        super(DAN_BPE, self).__init__()
        self.embedding = embedding_layer
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.long()
        embedded = self.embedding(x)
        averaged_embeds = torch.mean(embedded, dim=1)
        hidden = self.relu(self.fc1(averaged_embeds))
        logits = self.fc2(hidden)
        return logits