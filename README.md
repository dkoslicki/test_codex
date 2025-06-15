# raretarget
Linear algebraic method for identifying therapeutic intervention targets for rare diseases


### Dev info

Code has been confirmed to work with **Python 3.10.6** (have not tested other Python versions).

To get set up to run code:

1. Create and activate a python environment running Python 3.10
2. Run `pip install -r raretarget/requirements.txt`


### Repo overview

* [graphs](/graphs): Contains different toy/test graphs
* [orphanet](/orphanet): Contains the necessary gene/disease/symptom mappings from Orphanet, as well as scripts for generating those mappings
* [toy_implementation_jsonlines.py](/toy_implementation_jsonlines.py): David's human-readable initial toy implementation, adjusted to take in any specified JSON Lines formatted graph
  * Uses `SciPy.minimize` with limited-memory BFGS
  * Note: Contains an issue with generation of negative training examples, where not enough negative examples are generated for small graphs (fixed in Pytorch version)
* [pytorch_implementation.py](/pytorch_implementation.py): PyTorch implementation of our model; also contains a couple fixes over toy_implementation_jsonlines.py
  * Uses stochastic gradient descent (Adam)
  * Automatically detects whether to use `cuda`, `xps`, `mps`, or `cpu`
  * As of March 2025, only runs the first optimization, where it (globally) optimizes predicate weights, node weights, and the baseline offset
    * Second optimization is partially written, but commented out (not quite working); it may be worth circling back to this only after we get a better idea of how well the _first_ optimization is doing with its predictions, since this second step is expected to be more costly...and possibly not necessary? (TBD)
* [scratch](/scratch): Random scripts/work worth keeping around



