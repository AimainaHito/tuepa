Tüpa
====

**Tüpa** is a multilingual parser architecture for [Universal Conceptual Cognitive Annotation (UCCA)][1] based on the [TUPA](https://github.com/danielhers/tupa).

Requirements:

* tensorflow (tested on 1.12-gpu and 1.13-nightly-gpu)
* ELMoForManyLangs [available here](https://github.com/HIT-SCIR/ELMoForManyLangs).


ELMo-RNN
--------

Before training the elmo-rnn the training and validation data needs to be preprocessed. 

1. Preprocess the data with `preprocess_elmo.py`. 
2. Train the model with `train_elmo.py`
3. Evaluate the model with `evaluate_elmo.py` 

###### Usage of `preprocess_elmo.py`:

```
python tuepa/preprocess_elmo.py <path-to-train-files> <path-to-val-files> <train>.hdf5 <val>.hdf5 --save-dir save_dir -elmo <path-to-elmo>
```

###### Example usage of `train_elmo.py`:
 
 ```
 train.hdf5  val.hdf5 --layers '[{"activation": "relu", "neurons": 768,"updown": 1},{"activation": "relu", "neurons": 768,"updown": 1},{"activation": "relu", "neurons": 768,"updown": 1}]' -v --batch-size 128 -e 300 --learning-rate 0.001 --layer-dropout 0.85 --input-dropout 0.9 --save-dir save_dir --epoch_steps 100 --history-embedding-size 50 -top-rnn 1024 -hist-rnn 256
```

###### Usage of `evaluate_elmo.py`:

 ```
python tuepa/evaluate_elmo.py <model_dir> <path-to-eval-files> -elmo <path-to-elmo>
```

License
-------
This package is licensed under the GPLv3 or later license (see [`LICENSE.txt`](LICENSE.txt)).
The source code contains heavily modified code from the TUPA parser implementation by Daniel Hershcovich.
Its orginal source code can be retrieved from [this repository (version 1.3.7)](https://github.com/danielhers/tupa/releases/tag/v1.3.7).

Citations
---------

This project makes use of the parser architecture described in [the following paper](http://aclweb.org/anthology/P17-1104):

    @InProceedings{hershcovich2017a,
      author    = {Hershcovich, Daniel  and  Abend, Omri  and  Rappoport, Ari},
      title     = {A Transition-Based Directed Acyclic Graph Parser for UCCA},
      booktitle = {Proc. of ACL},
      year      = {2017},
      pages     = {1127--1138},
      url       = {http://aclweb.org/anthology/P17-1104}
    }

[1]: http://github.com/huji-nlp/ucca
