import numpy as np
import os
import scipy.io
import argparse
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import percentileofscore, norm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--roc', action='store_true', dest='roc')
parser.add_argument('--type', type=str, default='percentiles',
        choices=['percentiles', 'gaussian', 'threshold', 'logistic'])
parser.add_argument('--bias', type=float, default=1)
parser.add_argument('--showbest', action='store_true', dest='showbest')
parser.add_argument('--plot', action='store_true', dest='plot')
parser.set_defaults(roc=False, showbest=False, plot=False)
args = parser.parse_args()
os.chdir(args.dataset)

res101 = scipy.io.loadmat('res101.mat')
splits = scipy.io.loadmat('att_splits.mat')
labels = res101['labels'][:,0]
test_seen_loc = splits['test_seen_loc'][:,0]-1
test_unseen_loc = splits['test_unseen_loc'][:,0]-1
y_score = np.loadtxt('y_score.txt')
y_true = np.loadtxt('y_true.txt', dtype=int)

y_score_test = np.loadtxt('y_score_test.txt')
print(f'Val seen score mean: {y_score[y_true.astype(bool)].mean()}')
print(f'Test seen score mean: {y_score_test[:test_seen_loc.size].mean()}')
if args.plot:
    plt_val_s = sns.distplot(y_score[y_true.astype(bool)], hist=False, kde=True, label='val seen')
    plt_val_u = sns.distplot(y_score[~y_true.astype(bool)], hist=False, kde=True, label='val unseen')
    plt_test_s = sns.distplot(y_score_test[:test_seen_loc.size], hist=False, kde=True, label='test seen')
    plt_test_u = sns.distplot(y_score_test[test_seen_loc.size:], hist=False, kde=True, label='test unseen')
    np.savetxt('val_s_kde.txt', plt_val_s.axes.lines[0].get_xydata(), '(%1.4f, %1.4f)')
    np.savetxt('val_u_kde.txt', plt_val_u.axes.lines[1].get_xydata(), '(%1.4f, %1.4f)')
    np.savetxt('test_s_kde.txt', plt_test_s.axes.lines[2].get_xydata(), '(%1.4f, %1.4f)')
    np.savetxt('test_u_kde.txt', plt_test_u.axes.lines[3].get_xydata(), '(%1.4f, %1.4f)')
    plt.legend()
    plt.show()

y_true_test = np.append(
    np.ones(test_seen_loc.size, dtype=int),
    np.zeros(test_unseen_loc.size, dtype=int)
)

def novelty_gaussian(y_score, y_true, y_score_test, bias=1):
    mu_seen, std_seen = norm.fit(y_score[y_true.astype(bool)])
    mu_unseen, std_unseen = norm.fit(y_score[~y_true.astype(bool)])
    p_seen = norm.cdf(y_score_test, mu_seen, std_seen)
    p_unseen = 1-norm.cdf(y_score_test, mu_unseen, std_unseen)
    print(mu_seen, std_seen, mu_unseen, std_unseen)
    return bias*p_unseen/(p_seen+bias*p_unseen)

def novelty_percentiles(y_score, y_true, y_score_test, bias=1):
    yp_seen = np.array([
        percentileofscore(y_score[y_true.astype(bool)], x) for x in y_score_test
    ])/100
    yp_unseen = 1-np.array([
        percentileofscore(y_score[~y_true.astype(bool)], x) for x in y_score_test
    ])/100
    return bias*yp_unseen/(yp_seen+bias*yp_unseen)

def novelty_logistic(y_score, y_true, y_score_test, bias=1):
    model = LogisticRegression(
        multi_class='ovr',
        #class_weight='balanced',
        class_weight = {
            0: y_true.sum()*bias,
            1: (1-y_true).sum()
        }
    ).fit(y_score.reshape(-1,1), y_true)
    return model.predict_proba(y_score_test.reshape(-1,1))[:,0]

def class_avg_acc(scores, loc, seen, threshold):
    classes = np.unique(labels[loc])
    scores_by_class = [scores[labels[loc] == c] for c in classes]
    class_acc = [
        np.mean(cs > threshold if seen else cs <= threshold) for cs in scores_by_class
    ]
    return np.mean(class_acc)

def novelty_threshold_per_class(y_score, y_true, y_score_test, test_seen_loc,
    test_unseen_loc, bias=1):
    fpr,tpr,thresholds = roc_curve(y_true, y_score)
    seen_acc = np.asarray([
        class_avg_acc(y_score[y_true.astype(bool)], test_seen_loc, True, t)
        for t in thresholds
    ])
    unseen_acc = np.asarray([
        class_avg_acc(y_score[~y_true.astype(bool)], test_unseen_loc, False, t)
        for t in thresholds
    ])
    unseen_acc_biased = bias * unseen_acc
    idx = np.argmax(2*unseen_acc_biased*seen_acc/(unseen_acc_biased+seen_acc))
    threshold = thresholds[idx]
    return y_score_test <= threshold

def novelty_threshold(y_score, y_true, y_score_test, bias=1):
    fpr,tpr,thresholds = roc_curve(y_true, y_score)
    idx = np.argmax(tpr+(1-fpr)*bias)
    threshold = thresholds[idx]
    return y_score_test <= threshold

def print_acc(novelty_seen, novelty_unseen):
    novelty_seen_acc = (novelty_seen < 0.5).sum()/novelty_seen.size
    novelty_unseen_acc = (novelty_unseen > 0.5).sum()/novelty_unseen.size
    print(f'Seen acc: {novelty_seen_acc}') 
    print(f'Unseen acc: {novelty_unseen_acc}')

def print_acc_per_class(novelty_seen, novelty_unseen, seen_loc, unseen_loc):
    seen_classes = np.unique(labels[seen_loc])
    seen_acc = [np.mean(novelty_seen[labels[seen_loc] == c] < 0.5) for c in seen_classes]
    unseen_classes = np.unique(labels[unseen_loc])
    unseen_acc = [np.mean(novelty_unseen[labels[unseen_loc] == c] > 0.5) for c in unseen_classes]
    print(f'Seen acc (per class): {np.mean(seen_acc)}') 
    #print(seen_acc)
    print(f'Unseen acc (per class): {np.mean(unseen_acc)}')
    #print(unseen_acc)

if args.type == 'percentiles':
    novelty = novelty_percentiles(y_score, y_true, y_score_test, bias=args.bias)
    novelty_best = novelty_percentiles(y_score_test,y_true_test,y_score_test)
elif args.type == 'gaussian':
    novelty = novelty_gaussian(y_score, y_true, y_score_test, bias=args.bias)
    novelty_best = novelty_gaussian(y_score_test,y_true_test,y_score_test)
elif args.type == 'threshold':
    novelty = novelty_threshold(y_score, y_true, y_score_test, bias=args.bias)
    novelty_best = novelty_threshold(y_score_test, y_true_test, y_score_test)
elif args.type == 'logistic':
    novelty = novelty_logistic(y_score, y_true, y_score_test, bias=args.bias)
    novelty_best = novelty_logistic(y_score_test, y_true_test, y_score_test)
elif args.type == 'threshold-pc':
    train_seen_loc = np.loadtxt('train_seen_loc.txt', dtype=int)
    train_unseen_loc = np.loadtxt('train_unseen_loc.txt', dtype=int)
    novelty = novelty_threshold_per_class(y_score, y_true, y_score_test,
            train_seen_loc, train_unseen_loc, bias=args.bias)
    novelty_best = novelty_threshold_per_class(y_score_test, y_true_test,
            y_score_test, test_seen_loc, test_unseen_loc)
novelty_seen, novelty_unseen = np.split(novelty, [test_seen_loc.size])
print('TEST')
print(f'AUC: {roc_auc_score(y_true_test, y_score_test)}')
print_acc(novelty_seen, novelty_unseen)
print_acc_per_class(novelty_seen, novelty_unseen, test_seen_loc,
        test_unseen_loc)
print(f'Total acc: {((novelty_seen < 0.5).sum() + (novelty_unseen > 0.5).sum())/(novelty_seen.size+novelty_unseen.size)}')
np.savetxt(f'novelty_seen.txt', novelty_seen)
np.savetxt(f'novelty_unseen.txt', novelty_unseen)
if args.showbest:
    print('BEST:')
    print_acc(novelty_best[y_true_test.astype(bool)],
            novelty_best[~y_true_test.astype(bool)])
    print(f'Total acc: {((novelty_best[y_true_test.astype(bool)] < 0.5).sum() + (novelty_best[~y_true_test.astype(bool)] > 0.5).sum())/(novelty_seen.size+novelty_unseen.size)}')
if args.roc:
    roc = roc_curve(y_true, y_score)
    plt.plot(roc[0], roc[1])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()
