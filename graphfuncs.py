# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:11:03 2016

@author: mberman
"""

from skimage.future.graph import RAG
import numpy as np

import cv2

def rag_histograms(image, labels, connectivity=2, gt=None, bins=5, method='opencv'):
    # histogram distance question:
    # https://stackoverflow.com/questions/6499491/comparing-two-histograms
    # http://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
    """Compute the Region Adjacency Graph using histogram distance.
    ----------
    image : ndarray, shape(M, N, [..., P,] 3)
        Input image.
    labels : ndarray, shape(M, N, [..., P,])
        The labelled image. This should have one dimension less than
        `image`. If `image` has dimensions `(M, N, 3)` `labels` should have
        dimensions `(M, N)`.
    Returns
    -------
    out : RAG
        The region adjacency graph.
    """
    graph = RAG(labels, connectivity=connectivity)
    
    for n in graph:
        region = (labels == n)
        if method == 'opencv':
            hist = cv2.calcHist([image], [0, 1, 2], region.astype(np.uint8), [bins, bins, bins],
    		[0., 1., 0., 1., 0., 1.])
            cv2.normalize(hist, hist)
            hist = hist.flatten()
        elif method == 'channelcat':
            crop = image[region, :].reshape(-1, 3)
            red, rbins = np.histogram(crop[:, 0], bins=bins[0], density=True, range=(0., 1.))
            green, gbins = np.histogram(crop[:, 1], bins=bins[1], density=True, range=(0., 1.))
            blue, bbins = np.histogram(crop[:, 2], bins=bins[2], density=True, range=(0., 1.))
            hist = np.hstack((red, green, blue))
        elif method == 'channelcat2':
            crop = image[region, :].reshape(-1, 3)
            red = cv2.calcHist([image], [0], region.astype(np.uint8), [bins[0]], [0., 1.])
            green = cv2.calcHist([image], [1], region.astype(np.uint8), [bins[1]], [0., 1.])
            blue = cv2.calcHist([image], [2], region.astype(np.uint8), [bins[2]], [0., 1.])
            hist = np.vstack((red, green, blue))
            cv2.normalize(hist, hist)
            hist = hist.flatten()
        graph.node[n].update({'labels': [n],
                              'feat': hist})
        if gt is not None:
            # majority vote for superpixel label
            (values,counts) = np.unique(gt[region],return_counts=True)
            ind=np.argmax(counts)
            graph.node[n]['gt'] = values[ind]
        
    for x, y, d in graph.edges_iter(data=True):
        diff = graph.node[x]['feat'] - graph.node[y]['feat']
        diff = np.abs(diff)
        d['weight'] = diff

    return graph
    