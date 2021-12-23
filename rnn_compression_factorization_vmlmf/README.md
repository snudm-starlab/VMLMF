# VMLMF: Vector multiplication on Low-rank Matrix Factorization
This package provide implementations of VMLMF learning a compressed LSTM model.

## Overview 
- RNN compression based on Low-rank Matrix Factorization
    - Version 1.0
    - Last Updated 10.15.21

## Install
### Environment
- Ubuntu 16.04(LTS)
- CUDA 10.1
- Python 3.8
- PyTorch 1.7.1

## How to use
#### Code structure
```
VMLMF
  │ 
  ├── src
  │    │     
  │    ├── models
  │    │     ├── vmlmf_group.py: vmlmf with group structure cell and network code
  │    │     ├── vmlmf_lm.py: vmlmf for language model cell and network code
  │    │     └── vmlmf.py: vmlmf cell and network code
  │    │      
  │    └── train_test
  │    |     ├── main.py: control training and testing 
  │    |     ├── train.py: train the models on the Human Activity Recognition tasks 
  |    |     └── test.py: test the models on the Human Activity Recognition tasks 
  |    |  
  │    └── utils
  |           └── utilities for the package
  │    
  └── scripts: shell scripts for training and testing
```


#### Datasets description
*Opportunity dataset [[Homepage]](https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition)
*UCI dataset [[Homepage]](https://archive.ics.uci.edu/ml/datasets.php)
*Pen Tree Bank dataset [[Homepage]](https://deepai.org/dataset/penn-treebank)
    * Visit the official hompage to check the detail information.
    * You can download the datasets on the website.
   
## How to use 
#### Download the zip file
    cd VMLMF

#### Install the required packages
    install pytorch 1.7.1 proper to your environment  (1.7.1 is required!!)
    pip install -r requirements.txt
    
If other packages are required, use "pip install" to install them.

#### Download Datasets
    sh preprocess.sh

#### Run the demo
    sh ./script/demo.sh

#### Training & Evaluation
* You can test the models you want:
    ```    
    bash demo.sh
    ```
## Contact us
- Hyojin Jeon (tarahjjeon@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab. at Seoul National University.

*This software may be used only for research evaluation purposes.*
*For other purposes (e.g., commercial), please contact the authors.*
