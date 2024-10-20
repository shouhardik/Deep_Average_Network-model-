# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, read_word_embeddings
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW, SentimentDatasetDAN, SentimentDatasetBPE
from DANmodels import DAN, DAN_BPE
from BPEmodel import BytePairEncoding
from newBPE import Byte_PairEncoding

# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()
        # X = X.to(torch.int64)
        #X = X.float().to(torch.int64)
        #X = torch.tensor(model).to(torch.int64)
        #if (y < 0).any(): continue
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()
        # X = X.to(torch.int64)
        #X = X.float().to(torch.int64)
        #X = torch.tensor(model).to(torch.int64)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy

# Experiment function to run training and evaluation for multiple epochs
def experimentOptimizer(model, train_loader, test_loader, optimizer_type, lr, momentum=0.9):
    # Loss function
    loss_fn = nn.NLLLoss()
    
    # Optimizer: Adam or SGD with momentum
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-5)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    all_train_accuracy = []
    all_test_accuracy = []
    
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load dataset
    start_time = time.time()

    train_data = SentimentDatasetBOW("data/train.txt")
    dev_data = SentimentDatasetBOW("data/dev.txt")
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data loaded in : {elapsed_time} seconds")


    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()
    elif args.model == "DAN":
        word_embeddings = read_word_embeddings("data/glove.6B.300d-relativized.txt")
        word_to_index_train = create_vocabulary("data/train.txt")
        word_to_index_dev = create_vocabulary("data/dev.txt")
        train_data = SentimentDatasetDAN("data/train.txt", word_embeddings)
        dev_data = SentimentDatasetDAN("data/dev.txt", word_embeddings)

        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=True)

        embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=True)
        #embedding_layer = word_embeddings.get_initialized_embedding_layerWithoutGlove(frozen=True)
        input_size=300
        hidden_size = 128
        dan_model = DAN(input_size, hidden_size, embedding_layer, dropout_rate=0.5)
        print("\nTraining Deep Averaging Network (DAN):")

        # Train and evaluate your DAN model using your experiment function
        dan_train_accuracy, dan_test_accuracy = experiment(
            dan_model, 
            train_loader, 
            test_loader
        )
        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_accuracy, label='DAN')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for DAN')
        plt.legend()
        plt.grid()
        training_accuracy_file = 'dan_train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_test_accuracy, label='DAN')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for DAN')
        plt.legend()
        plt.grid()
        dev_accuracy_file = 'dan_dev_accuracy.png'
        plt.savefig(dev_accuracy_file)
        print(f"Dev accuracy plot saved as {dev_accuracy_file}\n\n")

    elif args.model == "DAN-withoutGlove":
        word_embeddings = read_word_embeddings("data/glove.6B.300d-relativized.txt")
        word_to_index_train = create_vocabulary("data/train.txt")
        word_to_index_dev = create_vocabulary("data/dev.txt")
        train_data = SentimentDatasetDAN("data/train.txt", word_embeddings)
        dev_data = SentimentDatasetDAN("data/dev.txt", word_embeddings)

        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=True)

        #embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=True)
        embedding_layer = word_embeddings.get_initialized_embedding_layerWithoutGlove(frozen=True)
        input_size=300
        hidden_size = 128
        dan_model = DAN(input_size, hidden_size, embedding_layer, dropout_rate=0.5)
        print("\nTraining Deep Averaging Network (DAN) without GloVe:")

        # Train and evaluate your DAN model using your experiment function
        dan_train_accuracy, dan_test_accuracy = experiment(
            dan_model, 
            train_loader, 
            test_loader
        )
        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_accuracy, label='DAN')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for DAN')
        plt.legend()
        plt.grid()
        training_accuracy_file = 'dan_train_accuracy1.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_test_accuracy, label='DAN')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for DAN')
        plt.legend()
        plt.grid()
        dev_accuracy_file = 'dan_dev_accuracy1.png'
        plt.savefig(dev_accuracy_file)
        print(f"Dev accuracy plot saved as {dev_accuracy_file}\n\n")

    elif args.model == "SUBWORD":

        with open('data/train.txt', 'r') as file:
            corpus = file.read()

        bpe = BytePairEncoding(vocab_size = 8000)
        #bpe = BPE(corpus, 1000)

        bpe.train_bpe(corpus)
        #bpe_vocab = bpe.bpe_vocab
        bpe_vocab = bpe.get_vocab(corpus)
        #print(bpe_vocab)
        #bpe_vocab = bpe._build_vocab(corpus)
        
        # print("BPE Vocabulary Size:", len(bpe_vocab))
        # print("Sample tokens from vocabulary:", list(bpe_vocab.items())[:5])

        train_dataset = SentimentDatasetBPE('data/train.txt', bpe, bpe_vocab, max_len=100)
        encoded_sentence, label = train_dataset[0]
        decoded_sentence = train_dataset.decodeSentence(encoded_sentence)
        print("Original sentence:", train_dataset.sentences[0])
        print("Encoded sentence:", encoded_sentence)
        print("Decoded sentence:", decoded_sentence)
        dev_dataset = SentimentDatasetBPE('data/dev.txt', bpe, bpe_vocab, max_len=100)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
        #print('Length'+str(test_loader[:5]))
        word_embeddings = read_word_embeddings("data/glove.6B.300d-relativized.txt")
        embedding_dim = 300 
        embedding_layer = word_embeddings.get_initialized_embedding_layerWithoutGloveNew(len(bpe_vocab))
        # Hyperparameters
         # Example embedding size
        output_dim = 2  # Number of classes (e.g., 2 for positive/negative)

        # Initialize the DAN model
        vocab_size = len(bpe_vocab)  # Your BPE vocab size
        model = DAN_BPE(vocab_size, embedding_dim, embedding_layer, output_dim, 256)

        dan_train_accuracy, dan_test_accuracy = experimentOptimizer(
            model, 
            train_loader, 
            test_loader,
             "adam",
             0.001
        )
        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_accuracy, label='DAN')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for DAN with BPE')
        plt.legend()
        plt.grid()
        training_accuracy_file = 'dan_train_accuracy_bpe.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_test_accuracy, label='DAN')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for DAN with BPE')
        plt.legend()
        plt.grid()
        dev_accuracy_file = 'dan_dev_accuracy_bpe.png'
        plt.savefig(dev_accuracy_file)
        print(f"Dev accuracy plot saved as {dev_accuracy_file}\n\n")





def create_vocabulary(file_path, max_vocab_size=10000):
    word_counts = {}
    with open(file_path, 'r') as f:
        for line in f:
            _, sentence = line.strip().split('\t')
            for word in sentence.split():
                word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort words by frequency and take top max_vocab_size
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:max_vocab_size]
    
    # Create word_to_index dictionary
    word_to_index = {word: idx + 1 for idx, (word, _) in enumerate(sorted_words)}
    word_to_index['<UNK>'] = 0  # Add unknown token
    
    return word_to_index

if __name__ == "__main__":
    main()
