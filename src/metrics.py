# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 13:01:29 2021

@author: lauracue
"""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import precision_recall_curve


def find_best_f0_5_threshold(pred, gt, precision_constraint=0.70, recall_constraint=0.30):
    precision, recall, thresholds = precision_recall_curve(gt, pred)
    
    # Filter for thresholds that meet the constraints
    valid_indices = np.where((precision[:-1] >= precision_constraint) & (recall[:-1] >= recall_constraint))
    valid_thresholds = thresholds[valid_indices]
    
    if valid_thresholds.size > 0:
        # If there are valid thresholds, choose the one with the best F0.5 score
        valid_precisions = precision[valid_indices]
        valid_recalls = recall[valid_indices]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            F0_5_scores = 1.25 * valid_precisions * valid_recalls / (0.25 * valid_precisions + valid_recalls)
            # In case of a zero division, set the corresponding F0.5 score to zero
            F0_5_scores[np.isnan(F0_5_scores)] = 0
        
        best_idx = np.argmax(F0_5_scores)
        best_F0_5_score = F0_5_scores[best_idx]
        return valid_thresholds[best_idx], valid_precisions[best_idx], valid_recalls[best_idx], best_F0_5_score
    else:
        # If no valid thresholds, choose the best threshold based on F0.5 score without applying constraints
        with np.errstate(divide='ignore', invalid='ignore'):
            F0_5_scores = 1.25 * precision[:-1] * recall[:-1] / (0.25 * precision[:-1] + recall[:-1])
            F0_5_scores[np.isnan(F0_5_scores)] = 0
            
        best_idx = np.nanargmax(F0_5_scores)  # Use nanargmax to ignore NaN values
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
        best_F0_5_score = F0_5_scores[best_idx]
        return best_threshold, precision[best_idx], recall[best_idx], best_F0_5_score



def evaluate_metrics(pred, gt):
    accu_criteria = dict()
    if type(pred).__module__ != np.__name__:
        pred = pred.data.cpu().numpy()
    if type(gt).__module__ != np.__name__:
        gt = gt.data.cpu().numpy()
        
    best_threshold, precision, recall, best_f05 = find_best_f0_5_threshold(pred, gt)

    # Once the best threshold is found, calculate all metrics at this threshold
    pred_final = (pred >= best_threshold).astype(int)
    gt= gt.astype(int)
    accuracy = accuracy_score(gt, pred_final) * 100
    
    accu_criteria["Accuracy"] = np.round(accuracy,2)
    
    f1 = f1_score(gt, pred_final, average=None, zero_division=True)
    pre = precision_score(gt, pred_final, average=None, zero_division=True)
    rec = recall_score(gt, pred_final, average=None, zero_division=True)

    accu_criteria = {
        "Accuracy": np.round(accuracy, 2),
        "F1": np.round(np.array(f1)*100,2),
        "Pre": np.round(np.array(pre)*100,2),
        "Rec": np.round(np.array(rec)*100,2),
        "F0.5": np.round(np.array(best_f05)*100,2),
        "Best Threshold": best_threshold
    }

    return accu_criteria



def evaluate_metrics_pred(pred, gt, val = 0):
    accu_criteria = dict()
    if type(pred).__module__ != np.__name__:
        pred = pred.data.cpu().numpy()
    if type(gt).__module__ != np.__name__:
        gt = gt.data.cpu().numpy()

    c = pred.shape[1]

    pred = np.argmax(pred,axis=1)

    mask = np.where(gt>0)
    gt = gt[mask][:]
    pred = pred[mask]
    accuracy = accuracy_score(gt, pred)*100
    accu_criteria["Accuracy"] = np.round(accuracy,2)
    
    f1 = f1_score(gt, pred, average=None, zero_division=True)
    pre = precision_score(gt, pred, average=None, zero_division=True)
    rec = recall_score(gt, pred, average=None, zero_division=True)


    accu_criteria["avgF1"] = np.round(np.sum(f1[1:])*100/(c-1),2)
    accu_criteria["avgPre"] = np.round(np.sum(pre[1:])*100/(c-1),2)
    accu_criteria["avgRec"] = np.round(np.sum(rec[1:])*100/(c-1),2)
    accu_criteria["F1"] = np.round(np.array(f1[1:])*100,2)
    accu_criteria["Pre"] = np.round(np.array(pre[1:])*100,2)
    accu_criteria["Rec"] = np.round(np.array(rec[1:])*100,2)

    return accu_criteria