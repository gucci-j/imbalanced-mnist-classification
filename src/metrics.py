import torch
import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

import matplotlib.pyplot as plt
import seaborn as sns


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)

    return acc


def compute_f1_prec_rec(preds: np.array, y: np.array):
    rounded_preds = (preds >= 0.5).astype(int)
    f1 = f1_score(y, rounded_preds)
    prec = precision_score(y, rounded_preds)
    rec = recall_score(y, rounded_preds)

    return f1, prec, rec


def draw_cm(preds: np.array, y: np.array, run_start_time: str):
    rounded_preds = (preds >= 0.5).astype(int)
    cm_data = confusion_matrix(y, rounded_preds) # (tn, fp, fn, tp)
    normalised_cm_data = cm_data.astype(float) / cm_data.sum(axis=1)[:, np.newaxis]
    
    # plot normal one
    plt.figure()
    sns.set_style()
    sns.set_context("paper")
    sns.heatmap(cm_data, annot=True, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig('./fig/{}/cm.png'.format(run_start_time), bbox_inches="tight", pad_inches=0.1)
    plt.clf()

    # plot normalised one
    sns.heatmap(normalised_cm_data, annot=True, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalised Confusion Matrix")
    plt.savefig('./fig/{}/normalised_cm.png'.format(run_start_time), bbox_inches="tight", pad_inches=0.1)



def draw_roc(preds: np.array, y: np.array, run_start_time: str) -> float:
    fpr_list, tpr_list, thresh_list = roc_curve(y, preds)
    auc = roc_auc_score(y, preds)

    plt.figure()
    plt.plot(fpr_list, tpr_list)
    plt.grid(True)
    plt.title(f"ROC Curve (AUC={auc:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig('./fig/{}/roc.png'.format(run_start_time), bbox_inches="tight", pad_inches=0.1)

    return auc


def draw_prc(preds: np.array, y: np.array, run_start_time: str) -> float:
    prec_list, rec_list, thresh_list = precision_recall_curve(y, preds)
    ap = average_precision_score(y, preds)

    plt.figure()
    plt.plot(rec_list, prec_list)
    plt.grid(True)
    plt.title(f"PR Curve (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig('./fig/{}/pr.png'.format(run_start_time), bbox_inches="tight", pad_inches=0.1)

    return ap


def draw_det(preds: np.array, y: np.array, run_start_time: str) -> float:
    thresh_list = np.arange(0.01, 1.0, 0.01)
    fpr_list = []
    fnr_list = []

    for thresh in thresh_list:
        threshed_preds = (preds >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, threshed_preds).ravel() # (tn, fp, fn, tp)
        fpr_list.append(fp / (tn + fp))
        fnr_list.append(fn / (tp + fn))
    
    eer = estimate_eer(fpr_list, fnr_list)

    plt.figure()
    plt.plot(fpr_list, fnr_list)
    plt.plot(eer, eer, color="red", marker="x")
    plt.grid(True)
    plt.title(f"DET Curve (EER={eer:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("False Negative Rate")
    plt.savefig('./fig/{}/det.png'.format(run_start_time), bbox_inches="tight", pad_inches=0.1)

    return eer


def estimate_eer(fpr_list: list, fnr_list: list) -> float:
    best_diff = None
    eer = None
    for fpr, fnr in zip(fpr_list, fnr_list):
        diff = abs(fpr - fnr)
        if best_diff is None: # init
            best_diff = diff
            eer = fpr
        elif diff < best_diff: # update
            best_diff = diff
            eer = fpr
    
    return eer