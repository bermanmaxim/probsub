# probsub.py
# General probably submodular learning interface

from __future__ import absolute_import, division, print_function  # We require Python 2.6 or later

from tqdm import tqdm
# packages for SLIC superpixels
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import graphfuncs
import numpy as np

from one_slack_ssvm_hard import OneSlackSSVMHard

params = {
    'slic': {
        'segments': 500, # number of superpixels
        'sigma': 3,
    },
    'ssvm':{
        'C': 1.,
    },
}

import os
try:
    import cPickle as pickle
except:
    import pickle

def segments(im, params_slic):
    return slic(im, n_segments = params_slic['segments'], sigma = params_slic['sigma']) + 1 # so min. label is 1

def build_graph(params, dataset, example, memoize=False, load=True):
    """
    add feature graph to an example of the dataset
    """
    params_slic = params['slic']

    def do_it():
        im = dataset['helper'].get_image(dataset, example)
        segments = segments(im, params_slic)
        if 'mask' in example:
            m = dataset['helper'].get_mask(dataset, example)
            graph = graphfuncs.rag_histograms(im, segments, gt=m)
        else:
            graph = graphfuncs.rag_histograms(im, segments)

        x_pystruct_features = np.vstack([graph.node[n]['feat'] for n in graph])
        x_pystruct_edges = np.array(graph.edges()) - 1
        x_pystruct_edgefeatures = np.vstack([graph.edge[n1][n2]['weight'] for (n1, n2) in graph.edges_iter()])

        example['x'] = (x_pystruct_features, x_pystruct_edges, x_pystruct_edgefeatures)
        if 'mask' in example:
            example['y'] = np.array([graph.node[n]['gt'] for n in graph], dtype=int)
        return example

    if memoize is False:
        return do_it()
    else:
        cachef = os.path.join(memoize, example["name"] + ".pkl")
        lock = cachef + ".lock"
        if os.path.isfile(lock):
            return "locked"
        else:
            try:
                if load:
                    example_cached = pickle.load(open(cachef, "rb"))
                    example.clear()
                    example.update( t for t in example_cached.viewitems() ) # change in-place!
                    return example
                else:
                    open(cachef, "rb").close()
                    return "exists"
            except (KeyboardInterrupt, SystemExit):
                raise
            except: # actually do the computation
                try:
                    open(lock, 'a').close()
                    do_it()
                    pickle.dump(example, open(cachef, "wb"), pickle.HIGHEST_PROTOCOL)
                    return example
                finally:
                    os.remove(lock)

    


def build_graphs(params, dataset, examples=-1, memoize=False, load=True):
    """
    add feature graphs to examples of the dataset
    if examples = -1, apply to all dataset['loaded_examples']
    """
    if examples == -1:
        examples = dataset['loaded_examples']
    for example in tqdm(examples, desc='Building graphs'):
        build_graph(params, dataset, example, memoize, load)
    return examples

def binarize(dset, examples, cat, copy=True):
    c = dset["helper"].labels.index(cat)
    if copy:
        examples = deepcopy(examples)
    for ex in tqdm(examples, desc='Binarizing masks'):
        ex['y_src'] = ex['y'].copy()
        ex['class'] = cat
        ex['y'][ex['y'] != c] = 0
        ex['y'][ex['y'] == c] = 1
    return examples

def get_constraints(constraints=()):
    if constraints == ():
        return {'type': 'empty'}
    else:
        cset, cex = constraints
        cdict = {'type': cset}
        n_states = cdict['n_states'] = len(set(l for ex in cex for l in ex['y']))
        ufeatdim = cdict['ufeatdim'] = cex[0]['x'][0].shape[1] # graph.node[next(iter(graph))]['feat'].shape[0]
        efeatdim = cdict['efeatdim'] = cex[0]['x'][2].shape[1]
        jfeatdim = cdict['jfeatdim'] = n_states * ufeatdim + n_states * n_states * efeatdim
        pairj = cdict['pairj'] = n_states * ufeatdim
        # negative indices
        cdict['negind'] = np.ogrid[pairj + efeatdim : pairj + 3*efeatdim]    
        # positive indices
        cdict['posind'] = np.hstack((np.ogrid[pairj : pairj + efeatdim],
                                     np.ogrid[pairj + 3*efeatdim:jfeatdim]))
        cdict['examples'] = cex
        return cdict

import one_slack_ssvm_hard
import edge_features_loss_graph
SSVM = one_slack_ssvm_hard.OneSlackSSVMHard
CRF = edge_features_loss_graph.EdgeFeaturesLossGraph
import probsub_helpers

from copy import deepcopy

def get_learner(params, cdict, warmstartfrom=None):
    paramsssvm = params['ssvm']
    crf = CRF(inference_method=('ogm', {'alg': 'gc'})) # graph-cuts in OpenGM
    if cdict['type'] == 'empty':
        negative = positive = []
        initialize = generate = None
    elif cdict['type'] == 'C0':
        negative = positive = np.hstack((cdict['negind'], cdict['posind']))
        initialize = generate = None
    elif cdict['type'] == 'C1':
        negative = np.hstack((cdict['negind'], cdict['posind']))
        positive = cdict['posind']
        initialize = generate = None
    elif cdict['type'] == 'C2':
        negative = cdict['negind']
        positive = cdict['posind']
        initialize = generate = None
    elif cdict['type'] == 'C3':
        negative = positive = []
        initialize, generate = probsub_helpers.get_helpers(params, cdict)
    elif cdict['type'] == 'C4':
        negative = positive = []
        initialize, generate = probsub_helpers.get_helpers(params, cdict)
    if warmstartfrom is not None:
        learner = deepcopy(warmstartfrom)
        learner.hard_satisfied = False
        learner.converged = False
        learner.negativity_constraint = negative
        learner.positivity_constraint = positive
        learner.initialize_constraints = initialize
        learner.generate_hard_constraints = generate
    else:
        learner = SSVM(crf, inference_cache=0, C=paramsssvm['C'], tol=paramsssvm['tol'], max_iter=paramsssvm['max_iter'],
                n_jobs=1, negativity_constraint=negative, positivity_constraint=positive,
                initialize_constraints=initialize, generate_hard_constraints=generate,
                show_loss_every=params['log']['loss_every'], check_constraints=True, 
                break_on_bad=paramsssvm['break_on_bad'], verbose=params['log']['SSVM_verbose'])
    return learner

def fit(examples, constraints, params, startfrom = []):
    if startfrom:
        fitted = copy.deepcopy(startfrom)
    else:
        fitted = []
    if isinstance(constraints, tuple): # 1 constraint set
        constraints = [constraints]
    X = [ex['x'] for ex in examples]
    Y = [ex['y'] for ex in examples]
    for constraint in constraints:
        cdict = get_constraints(constraints=constraint)
        if not fitted:
            learner = get_learner(params, cdict)
            learner.fit(X, Y)
        else:
            learner = get_learner(params, cdict, warmstartfrom=fitted[-1])
            learner.fit(X, Y, warm_start=True)
        fitted.append(learner)
    return fitted





