# Deep_Average_Network_Model

**Description**  
This project implements a Deep Averaging Network (DAN) for sentiment classification along with techniques like Byte Pair Encoding (BPE) and Skip-Gram for word embeddings. The model is designed to classify sentiment using pre-trained word embeddings, and variations of DAN with and without GloVe embeddings.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Dependencies](#dependencies)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Models](#models)
6. [Contributing](#contributing)
7. [License](#license)

---

## Project Structure

```plaintext
├── sentiment_data.py        # Handles data reading and defines SentimentExample and WordEmbeddings
├── utils.py                 # Contains Indexer class for mapping indices and words in vocabulary
├── BOWmodels.py             # Implements Bag-of-Words models for sentiment classification
├── DANmodels.py             # Placeholder for Deep Averaging Network (DAN) model implementation
├── BPEmodel.py              # Placeholder for Byte Pair Encoding implementation
├── main.py                  # Script for model evaluation (Bag-of-Words and DAN models)
└── README.md                # Project overview and instructions
├── dan_train_accuracy.png   # Image file for DAN Train Accuracy for 1.A
├── dan_dev_accuracy.png     # Image file for DAN Dev Accuracy for 1.A
├── dan_dev_accuracy1.png     # Image file for DAN Train Accuracy for 1.B
├── dan_dev_accuracy1.png     # Image file for DAN Dev Accuracy for 1.B
├── dan_train_accuracy_bpe.png # Image file for BPE Train Accuracy for 2.A
├── dan_test_accuracy_bpe.png  # Image file for BPE Test Accuracy for 2.A
```
• sentiment data.py handles data reading. This also defines a SentimentExample object, which wraps a list of words with an integer label ( 0/1 ), as     well as a WordEmbeddings object, which contains pre-trained word embeddings for this dataset.
• utils.py implements an Indexer class, which can be used to maintain a mapping between indices and words in the vocabulary.
• BOWmodels.py contains simple discrete Bag-of-Words models for the sentiment classification task on the provided dataset. You can use this as a       
  reference for implementing your Deep Averaging Network (DAN) in part 1.
• DANmodels.py an empty file where you will implement your DAN models.
• main.py shows how to evaluate the Bag-of-Words models on the sentiment classification task.
You can use this as a reference for evaluating your DAN models.

## Dependencies

- **Python 3.6+**
- **PyTorch**
- **NumPy**
- **Anaconda**

## Installation
git clone [https://github.com/shouhardik/Deep_Average_Network-model-](https://github.com/shouhardik/Deep_Average_Network-model-.git)

## Usage

To run the models, use the following commands:

- **1.A DAN model**: python main.py --model DAN

- **1.B DAN model without GloVe embeddings**: python main.py --model DAN-withoutGlove

- **2.A BPE model**: python main.py --model SUBWORD

## Models

- **DAN**: Deep Averaging Network with pre-trained GloVe embeddings
- **DAN-withoutGlove**: Deep Averaging Network without pre-trained embeddings
- **BPE**: Subword model using Byte Pair Encoding (BPE)
  
## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
