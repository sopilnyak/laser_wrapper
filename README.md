# A Python wrapper for [LASER](https://github.com/facebookresearch/LASER/) (Language-Agnostic SEntence Representations) multilingual embeddings

A simple Python class for using LASER multilingual embeddings without explicitly running any bash scripts.

## Installation

Firstly, install LASER and its dependencies from the [original repository](https://github.com/facebookresearch/LASER/).
```
!git clone https://github.com/facebookresearch/LASER.git
!cd LASER && export LASER="<full path to LASER installation root>" && bash ./install_external_tools.sh && bash ./install_models.sh
```

Then clone the wrapper repository.
```
!git clone https://github.com/sopilnyak/laser_wrapper.git
```

## Example

```python
from laser_wrapper.laser import Laser

laser = Laser(encoder_path='LASER/models/bilstm.93langs.2018-12-26.pt', bpe_codes='LASER/models/93langs.fcodes')
laser(['A dog went for a walk.'])
# Will return a numpy array of shape (num_sentences, embedding_dim)
```

You can run the full example in a [notebook on Google Colaboratory](https://colab.research.google.com/drive/19Qs_84dVV7z_RvP8DErHioaguQAm8_r2).

## Documentation

```python
class Embedder(encoder_path, bpe_codes, use_gpu=False)
```
* **encoder_path:** path to encoder weights
* **bpe_codes:** path to file with BPE codes
* **use_gpu:** use GPU instead of CPU

```python
def Embedder.__call__(sentences, tokenizer_lang='en', verbose=False)
```
* **sentences:** list of strings with input sentences
* **tokenizer_lang:** language to perform tokenization with
* **verbose:** show log messages
* **return:** numpy array of shape (n_sentences, embedding_dim)
