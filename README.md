Tüpa
====

**Tüpa** is a multilingual parser architecture for [Universal Conceptual Cognitive Annotation (UCCA)][1] based on the [TUPA](https://github.com/danielhers/tupa).

Requirements:

* tensorflow (tested on 1.12-gpu)
* ELMoForManyLangs [available here](https://github.com/HIT-SCIR/ELMoForManyLangs).
* finalfrontier [available here](https://github.com/danieldk/finalfrontier-python) (english pretrained available [here](https://drive.google.com/file/d/1S2pllHdR81o4DrhKa_uA2YMXwO3I6Y5D/view?usp=sharing))
* Rust with the nightly toolchain installed. (tested on `nightly-2018-12-24-x86_64-unknown-linux-gnu`)

Issues
------
The compilation of finalfrontier-python fails due to some api changes in pyo3. 

Adding the following lines to the imports in `finalfrontier-python/src/lib.rs` will fix the issue.

```rust
use pyo3::PyIterProtocol;
use pyo3::PyObjectProtocol;
use pyo3::exceptions as exc;
```

The next step is to compile the library using `cargo +nigthly build --release`. 
This will output `libfinalfrontier.so` to 
`finalfrontier-python/target/release/`. Rename `libfinalfrontier.so` to `finalfrontier` and copy it to `tuepa/tuepa/`.

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
