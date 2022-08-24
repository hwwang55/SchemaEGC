# SchemaEGC

This repository is the PyTorh implementation of SchemaEGC ([arXiv](https://arxiv.org/abs/2206.02921)):
> Schema-Guided Event Graph Completion  
Hongwei Wang, Zixuan Zhang, Sha Li, Jiawei Han, Yizhou Sun, Hanghang Tong, Joseph P. Olive, Heng Ji  
arXiv preprint


### Files in the folder

- `data/`
  - `pandemic/` (Pandemic dataset)
  - `schema_IED.json` (Car-IED and General-IED schemas)
  - `schema_pandemic.json` (Disease-Outbreak schema)
- `src/`: implementation of SchemaEGC

__Note__: The three IED-related datasets are too big to be included in this repository.
Please download and unzip the three datasets from [here](https://drive.google.com/file/d/1Aem6z7OgQ_EN-Ye8Spg6aO3MbufUNoOE/view?usp=sharing), and put the three dataset directories under `data/`.



### Running the code

```
$ python main.py
```


### Required packages

The code has been tested running under Python 3.7, with the following packages installed (along with their dependencies):

- torch == 1.8.1
- dgl == 0.6.1
