# Probsub
Structured SVM with probably submodular constraints

## Citation
For use, please cite:
> Berman, Maxim, & Blaschko, Matthew B. (2016, December). Efficient optimization for probably submodular constraints in CRFs. In Proceedings of the NIPS workshop on constructive machine learning.

See also prior reference:
> Zaremba, Wojciech, & Blaschko, Matthew B. (2016, March). Discriminative training of CRF models with probably submodular constraints. In 2016 IEEE Winter Conference on Applications of Computer Vision (WACV). IEEE.

## Installation
* Create environment and install requirements
   ```shell
   conda create --name probsubenv python=2.7 numpy cvxopt scikit-image
   conda install -c https://conda.binstar.org/menpo opencv
   source activate probsubenv
   pip install git+https://github.com/bermanmaxim/pystruct.git@hardconstraints
   ```

   *Note* As indicated pystruct has to be fetched from the `weightedloss` branch of [my fork of pystruct](https://github.com/bermanmaxim/pystruct.git).

* install [opengm](https://github.com/opengm/opengm) with `Python` and `graph-cuts` extensions and and make the python module available in the environment (for graph-cut inference)

## Usage
* `one_slack_ssvm_hard.py` constains one-slack SSVM learner class for `pystruct` with additional hard constraints <w, a> >= b, either specified or generated with `learner.generate_hard_constraints`
* `probsub.py` provides an interface to learners with probably submodular constraints, provided by `probsub_helpers.py`

