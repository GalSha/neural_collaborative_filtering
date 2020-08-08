'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
'''
import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time
#from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None
_K_recall = None
_K_precision = None
_K_max = None
_num_items = None

def evaluate_model(model, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
        
    hits, ndcgs = [],[]
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for idx in range(len(_testRatings)):
        (hr,ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)      
    return (hits, ndcgs)

def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype='int32')
    predictions = _model.predict([users, np.array(items)],
                                 batch_size=100, verbose=0)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()

    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0

def evaluate_model_recall_precision(model, num_items, testRatings, K_recall, K_precision, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _K_recall
    global _K_precision
    global _K_max
    global _num_items
    _model = model
    _testRatings = testRatings
    _K_recall = K_recall
    _K_precision = K_precision
    _K_max = max(_K_precision,_K_recall)
    _num_items = num_items

    recalls, precisions = [], []
    if (num_thread > 1):  # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_recall_precision, range(len(_testRatings)))
        pool.close()
        pool.join()
        recalls = [r[0] for r in res]
        precisions = [r[1] for r in res]
        return (recalls, precisions)
    # Single thread
    for idx in range(len(_testRatings)):
        (recall, precision) = eval_recall_precision(idx)
        recalls.append(recall)
        precisions.append(precision)
    return (recalls, precisions)

def eval_recall_precision(idx):
    rating = _testRatings[idx]
    items = np.arange(_num_items)
    u = rating[0]
    true_items = np.array(rating[1:])
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype='int32')
    predictions = _model.predict([users, items],
                                 batch_size=100, verbose=0)
    for i in range(len(items)):
        map_item_score[items[i]] = predictions[i]

    # Evaluate top rank list
    ranklist_max = heapq.nlargest(_K_max, map_item_score, key=map_item_score.get)
    precision = getPrecision(ranklist_max[:_K_precision], true_items)
    recall = getRecall(ranklist_max[:_K_recall], true_items)
    return (precision, recall)

def getPrecision(ranklist_per, true_items):
    return float(np.intersect1d(ranklist_per,true_items).size) / _K_precision

def getRecall(ranklist_rec, true_items):
    return float(np.intersect1d(ranklist_rec,true_items).size) / min(true_items.size,_K_recall)
