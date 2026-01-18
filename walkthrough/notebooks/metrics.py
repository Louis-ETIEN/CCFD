import numpy as np

def classify(probas, threshold=0.5):
    return [0 if proba < threshold else 1 for proba in probas]

def compute_MME(true_classes, predicted_classes):
    n = len(true_classes)
    false_preds = 0
    for i in range(n):
        if true_classes[i] != predicted_classes[i]:
            false_preds += 1
    return false_preds / n

def compute_accuracy_score(true_classes, predicted_classes):
    return 1 - compute_MME(true_classes, predicted_classes)

def compute_TPR(true_classes, predicted_classes):
    num = 0
    den = 0
    n = len(true_classes)
    for i in range(n):
        if true_classes[i]:
            den += 1
            if predicted_classes[i]:
                num += 1
    return num / den if den else 1

def compute_TNR(true_classes, predicted_classes):
    num = 0
    den = 0
    n = len(true_classes)
    for i in range(n):
        if not true_classes[i]:
            den += 1
            if not predicted_classes[i]:
                num += 1
    return num / den if den else 1

def compute_FNR(true_classes, predicted_classes):
    return 1 - compute_TPR(true_classes, predicted_classes)

def compute_FPR(true_classes, predicted_classes):
    return 1 - compute_TNR(true_classes, predicted_classes)

def compute_BER(true_classes, predicted_classes):
    return (compute_FPR(true_classes, predicted_classes) + compute_FNR(true_classes, predicted_classes)) / 2

def compute_G_mean(true_classes, predicted_classes):
    return np.sqrt(compute_TPR(true_classes, predicted_classes) * compute_TNR(true_classes, predicted_classes))

def compute_precision(true_classes, predicted_classes):
    num = 0
    den = 0
    for i in range(len(true_classes)):
        if predicted_classes[i]:
            den += 1
            if true_classes[i]:
                num += 1
    return num / den if den else 1

def compute_NPV(true_classes, predicted_classes):
    num = 0
    den = 0
    for i in range(len(true_classes)):
        if not predicted_classes[i]:
            den += 1
            if not true_classes[i]:
                num += 1
    return num / den if den else 1

def compute_FDR(true_classes, predicted_classes):
    return 1 - compute_precision(true_classes, predicted_classes)

def compute_FOR(true_classes, predicted_classes):
    return 1 - compute_NPV(true_classes, predicted_classes)

def compute_F1_score(true_classes, predicted_classes):
    return (2 * compute_precision(true_classes, predicted_classes) * compute_TPR(true_classes, predicted_classes)) / (compute_precision(true_classes, predicted_classes) + compute_TPR(true_classes, predicted_classes))

def compute_ROC(true_classes, probas, thresholds):
    res = []
    for threshold in thresholds:
        res.append([compute_TPR(true_classes, classify(probas, threshold)), compute_FPR(true_classes, classify(probas, threshold))])
    return res

def compute_PR(true_classes, probas, thresholds):
    res = []
    for threshold in thresholds:
        res.append((compute_precision(true_classes, classify(probas, threshold)), compute_TPR(true_classes, classify(probas, threshold))))
    return res

def compute_AUC_ROC(ROC):
    """
    ROC is a list of (TPR, FPR) tuples for different thresholds
    """

    n = len(ROC)
    if n <= 1:
        raise ValueError("Minimum of two Values required to compute AUC")
    else:
        AUC = 0
        for i in range(n - 1):
            AUC += (ROC[i][0] + ROC[i+1][0]) / 2 * (ROC[i+1][1] - ROC[i][1])
        return -AUC

def compute_AUC_PR(PR):
    """
    ROC is a list of (TPR, FPR) tuples for different thresholds
    """

    n = len(PR)
    if n <= 1:
        raise ValueError("Minimum of two Values required to compute AUC")
    else:
        AUC = 0
        for i in range(n - 1):
            AUC += (PR[i][0] + PR[i+1][0]) / 2 * (PR[i+1][1] - PR[i][1])
        return -AUC

def estimate_costs(FN_cost, FP_cost, labels, probas):
    costs = [0] * 1001
    for i in range(len(probas)):
        if labels[i] == 0:
            for j in range(int(np.floor(probas[i] * 1000)) + 1):
                 costs[j] += FP_cost
        else:
            for j in range(int(np.ceil(probas[i] * 1000)), 1001):
                costs[j] += FN_cost
    return costs

def recalls_at_k(labels, probas, k):
    labels = np.array(labels)
    total_recall = sum(labels)
    recalls = []
    indexes = np.argsort(probas)
    for i in range(1, k+1):
        recalls.append(sum(labels[indexes[-i:]])/total_recall)
    return recalls

def precisions_at_k(labels, probas, k):
    labels = np.array(labels)
    precisions = []
    indexes = np.argsort(probas)
    for i in range(1, k+1):
        precisions.append(sum(labels[indexes[-i:]])/i)
    return precisions

def plot_cost_function(FN_cost, FP_cost, labels, probas, ax, title="Cost against threshold"):
    thresholds = [0.001 * i for i in range(1001)]
    # costs = []
    # for threshold in thresholds:
    #     costs.append(estimate_cost(FN_cost, FP_cost, labels, probas, threshold))
    costs = estimate_costs(FN_cost, FP_cost, labels, probas)
    if ax != None:
        fs = 15
        ax.set_title(title, fontsize=fs)
        ax.set_xlim([-0.01, 1.01])
        
        ax.set_xlabel('threshold', fontsize=fs)
        ax.set_ylabel('cost', fontsize=fs)

        ax.plot(thresholds, costs, label = f'FP = {FP_cost}, FN = {FN_cost}')
        ax.legend(loc = 'lower right')
    return thresholds, costs


def plot_ROC(labels, probas, ax, title="Receiving Operating Characteristic", random=True):
    fs = 15
    ax.set_title(title, fontsize=fs)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    
    ax.set_xlabel('False Positive Rate', fontsize=fs)
    ax.set_ylabel('True Positive Rate', fontsize=fs)

    thresholds = [0.01 * i for i in range(101)]
    ROCs = compute_ROC(labels, probas, thresholds)

    ax.plot([ROC[1] for ROC in ROCs], [ROC[0] for ROC in ROCs], color='blue', label = 'AUC ROC Classifier = {0:0.3f}'.format(compute_AUC_ROC(ROCs)))
    ax.legend(loc = 'lower right')

    if random:
        ax.plot([0, 1], [0, 1],'r--',label="AUC ROC Random = 0.5")

def plot_PR(labels, probas, ax, title="Precision/Recall", random=True):
    fs = 15
    ax.set_title(title, fontsize=fs)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    
    ax.set_xlabel('Recall', fontsize=fs)
    ax.set_ylabel('Precision', fontsize=fs)

    thresholds = [0.01 * i for i in range(101)]
    PRs = compute_PR(labels, probas, thresholds)

    ax.plot([PR[1] for PR in PRs], [PR[0] for PR in PRs], color='blue', label = 'AUC PR Classifier = {0:0.3f}'.format(compute_AUC_PR(PRs)))
    ax.legend(loc = 'lower right')

    if random:
        floor = sum(labels)/len(labels)
        ax.plot([0, 1], [floor, floor],'r--',label="AUC PR Random = 0.5")
