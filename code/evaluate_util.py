import numpy as np 
import torch
import math
import time


def pre_ranking(user_feature, item_feature, train_dict, valid_dict, test_dict):
    '''prepare for the ranking: construct input data'''

    user_rank_feature = {}
    for userID in test_dict:
        his_items = train_dict[userID]
        features = []
        feature_values = []
        mask = []
        item_idx = list(item_feature.keys())
        for idx in range(len(item_idx)):
            itemID = item_idx[idx]
            if itemID in his_items: # set as -inf if it's in training set
                mask.append(-999.0)
            else:
                mask.append(0.0)
            features.append(np.array(user_feature[userID][0]+item_feature[itemID][0]))
            feature_values.append(np.array(user_feature[userID][1]+item_feature[itemID][1], dtype=np.float32))
            
        features = torch.tensor(features).cuda()
        feature_values = torch.tensor(feature_values).cuda()
        mask = torch.tensor(mask).cuda()
        user_rank_feature[userID] = [features, feature_values, mask]
    
    return user_rank_feature


def Ranking(model, valid_dict, test_dict, train_dict, item_feature, user_rank_feature,\
            batch_size, topN, return_pred=False):
    """evaluate the performance of top-n ranking by recall, precision, and ndcg"""
    user_gt_test = []
    user_gt_valid = []
    user_pred = []
    user_pred_dict = {}
    user_item_top1k = {}

    for userID in test_dict:
        features, feature_values, mask = user_rank_feature[userID]
        
        batch_num = len(item_feature)//batch_size
        item_idx = list(item_feature.keys())
        st, ed = 0, batch_size
        
        for i in range(batch_num):
            batch_feature = features[st: ed]
            batch_feature_values = feature_values[st: ed]
            batch_mask = mask[st: ed]

            prediction = model(batch_feature, batch_feature_values)
            prediction = prediction + batch_mask
            if i == 0:
                all_predictions = prediction
            else:
                all_predictions = torch.cat([all_predictions, prediction], 0)
                
            st, ed = st+batch_size, ed+batch_size
        
#         prediction for the last batch
        batch_feature = features[st:]
        batch_feature_values = feature_values[st:]
        batch_mask = mask[st:]

        prediction = model(batch_feature, batch_feature_values)
        prediction = prediction + batch_mask
        if batch_num == 0:
            all_predictions = prediction
        else:
            all_predictions = torch.cat([all_predictions, prediction], 0)
            
        user_gt_valid.append(valid_dict[userID])
        user_gt_test.append(test_dict[userID])
        _, indices = torch.topk(all_predictions, topN[-1])
        pred_items = torch.tensor(item_idx)[indices].cpu().numpy().tolist()
        user_item_top1k[userID] = pred_items
        user_pred_dict[userID] = all_predictions.detach().cpu().numpy()
        user_pred.append(pred_items)
            
    valid_results = computeTopNAccuracy(user_gt_valid, user_pred, topN)
    test_results = computeTopNAccuracy(user_gt_test, user_pred, topN)
            
    if return_pred: # used in the inference.py
        return valid_results, test_results, user_pred_dict, user_item_top1k
    return valid_results, test_results
    

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
    
    
def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []
    
    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
        
        precision.append(round(sumForPrecision / len(predictedIndices), 4))
        recall.append(round(sumForRecall / len(predictedIndices), 4))
        NDCG.append(round(sumForNdcg / len(predictedIndices), 4))
        MRR.append(round(sumForMRR / len(predictedIndices), 4))
        
    return precision, recall, NDCG, MRR



def print_results(loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None: 
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]]), 
                            '-'.join([str(x) for x in valid_result[2]]), 
                            '-'.join([str(x) for x in valid_result[3]])))
    if test_result is not None: 
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]]), 
                            '-'.join([str(x) for x in test_result[2]]), 
                            '-'.join([str(x) for x in test_result[3]])))

