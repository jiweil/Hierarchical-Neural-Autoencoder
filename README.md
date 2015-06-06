# A Hierarchical Neural Autoencoder for Paragraphs and Documents

Implementations of the three models presented in the paper "A Hierarchical Neural Autoencoder for Paragraphs and Documents" by Jiwei Li, Minh-Thang Luong and Dan Jurafsky.

## Requirements:
GPU 
matlab >= 2014b
memory >= 2GB



## Folders
Standard_LSTM: Standard LSTM Autoencoder

hier_LSTM: Hierarchical LSTM Autoencoder 

hier_LSTM_Attention: Hierarchical LSTM Autoencoder with Attention 

## DownLoad [Data]http://cs.stanford.edu/~bdlijiwei/data.tar)
- `dictionary`: vocabulary
- `train_permute.txt`: training data for standard Model. Each line corresponds to one document/paragraph
- `train_source_permute_segment.txt`: source training data for hierarchical Models. Each line corresponds to one sentence. An empty line starts a new document/sentence. Documents are reversed. 
- `test_source_permute_segment.txt`: target training data for hierarchical Model.


