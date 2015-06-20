# A Hierarchical Neural Autoencoder for Paragraphs and Documents

Implementations of the three models presented in the paper "A Hierarchical Neural Autoencoder for Paragraphs and Documents" by Jiwei Li, Minh-Thang Luong and Dan Jurafsky, ACL 2015

## Requirements:
GPU 

matlab >= 2014b

memory >= 4GB



## Folders
Standard_LSTM: Standard LSTM Autoencoder

hier_LSTM: Hierarchical LSTM Autoencoder 

hier_LSTM_Attention: Hierarchical LSTM Autoencoder with Attention 

## DownLoad [Data](http://cs.stanford.edu/~bdlijiwei/data.tar)
- `dictionary`: vocabulary
- `train_permute.txt`: training data for standard Model. Each line corresponds to one document/paragraph
- `train_source_permute_segment.txt`: source training data for hierarchical Models. Each line corresponds to one sentence. An empty line starts a new document/sentence. Documents are reversed. 
- `test_source_permute_segment.txt`: target training data for hierarchical Model.

Training roughly takes 2-3 weeks for standard models and 4-6 weeks for hierarchical models on a K40 GPU machine.


For any question or bug with the code, feel free to contact jiweil@stanford.edu

```latex
@article{li2015hierarchical,
    title={A Hierarchical Neural Autoencoder for Paragraphs and Documents},
    author={Li, Jiwei and Luong, Minh-Thang and Jurafsky, Dan},
    journal={arXiv preprint arXiv:1506.01057},
    year={2015}
}
```
