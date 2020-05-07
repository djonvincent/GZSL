import scipy.io
import numpy as np
import torch
import os
import argparse
from model import SegregationNetwork, constituency_loss
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--g', type=float, default=0.5)
parser.add_argument('--plotloss', action='store_true', dest='plotloss')
parser.add_argument('--validate', action='store_true', dest='validate')
parser.add_argument('--mode', choices=['test','val'], default='test')
parser.add_argument('--lr', type=float, default=0.001)
parser.set_defaults(plotloss=False, validate=False)
args = parser.parse_args()
os.chdir(args.dataset)

device = torch.device('cuda')
np.random.seed(123)
torch.manual_seed(123)

def load_val_loc(fn):
    loc = []
    with open(fn) as f:
        for l in f.readlines():
            loc.append(np.array([int(x) for x in l.split(' ')], dtype=int))
    return loc

res101 = scipy.io.loadmat('res101.mat')
splits = scipy.io.loadmat('att_splits.mat')
val_train_loc = load_val_loc('val_train_loc.txt')
val_test_seen_loc = load_val_loc('val_test_seen_loc.txt')
val_test_unseen_loc = load_val_loc('val_test_unseen_loc.txt')

features = res101['features'].transpose()
labels = res101['labels'][:,0]-1
trainval_loc = splits['trainval_loc'][:,0]-1
test_seen_loc = splits['test_seen_loc'][:,0]-1
test_unseen_loc = splits['test_unseen_loc'][:,0]-1

def get_membership_scores(N, loc, nclasses, class_prototypes):
    membership_scores = np.zeros(len(loc))
    second_largest = np.zeros(len(loc))
    for idx, i in enumerate(loc):
        xi = torch.zeros(nclasses,2048).to(device)
        xj = torch.zeros(nclasses,2048).to(device)
        for j in range(nclasses):
            xi[j] = torch.from_numpy(features[i]).float().to(device)
            xj[j] = torch.from_numpy(class_prototypes[j]).float().to(device)
        u = N(xi, xj, xi)
        #print(u.max(dim=1).values)
        membership_scores[idx] = u.max(dim=1).values.mean().item()
        second_largest[idx] = torch.topk(u, 2, dim=1, sorted=True).values[:,1].mean().item()
    #print(f'2nd largest mean: {second_largest.mean()}')
    return membership_scores

def train_eval(train_loc, test_seen_loc, test_unseen_loc, lr=args.lr,
        batch_size=512, epochs=args.epochs, val=args.validate,
        plotloss=args.plotloss, num_train_scores=0):
    print(f'Train samples: {len(train_loc)}')
    print(f'Test seen samples: {len(test_seen_loc)}')
    print(f'Test unseen samples: {len(test_unseen_loc)}')
    val_seen_loc = np.random.choice(test_seen_loc, 500, replace=False)
    val_unseen_loc = np.random.choice(test_unseen_loc, 500, replace=False)
    seen_classes = np.unique(labels[train_loc])
    nclasses = len(seen_classes)
    unseen_classes = np.unique(labels[test_unseen_loc])
    print(f'Seen classes: {nclasses}')
    print(seen_classes)
    print(f'Unseen classes: {len(unseen_classes)}')
    print(unseen_classes)
    # Find mean features for each training class
    class_prototypes = np.zeros((nclasses, 2048))
    for i, c in enumerate(seen_classes):
        f = features[train_loc[labels[train_loc] == c]]
        class_prototypes[i] = f.mean(0)

    N = SegregationNetwork(nclasses).to(device)
    optim = torch.optim.Adam(N.parameters(), lr=lr)
    epoch = 0
    loss_arr = np.zeros(0)
    loss_arr_mean = np.zeros(0)
    plt.ion()
    plt.show()
    while epoch < epochs:
        print(f"Epoch: {epoch+1}")
        N.train()
        for b in range(40):
            optim.zero_grad()
            alpha = torch.clamp(torch.randn((batch_size,1)),-2,2).to(device)*0.25 + 0.5
            loc = np.random.choice(train_loc, size=(batch_size,2), replace=False)
            xi = torch.from_numpy(features[loc[:,0]]).float().to(device)
            xj = torch.from_numpy(features[loc[:,1]]).float().to(device)
            xk = xi*alpha + xj*(1-alpha)
            beta = torch.zeros(batch_size, nclasses).to(device)
            for i in range(batch_size):
                idx1 = np.argwhere(seen_classes == labels[loc[i,0]])[0][0]
                idx2 = np.argwhere(seen_classes == labels[loc[i,1]])[0][0]
                beta[i, idx1] = alpha[i][0]
                beta[i, idx2] = 1-alpha[i][0]

            u = N(xi, xj, xk)
            loss = constituency_loss(u, beta, args.g)
            loss.backward()
            optim.step()

            loss_arr = np.append(loss_arr, loss.item())
            loss_arr_mean = np.append(loss_arr_mean, loss_arr.mean())
        print(f'    Loss: {loss_arr[-40:].mean()}')
        print(f'    Mean loss: {loss_arr_mean[-1]}')

        if val:
            N.eval()
            with torch.set_grad_enabled(False): 
                seen_scores = get_membership_scores(N, val_seen_loc, nclasses,
                        class_prototypes)
                unseen_scores = get_membership_scores(N, val_unseen_loc, nclasses,
                        class_prototypes)
                y_true = np.append(np.ones(len(seen_scores)), np.zeros(len(unseen_scores)))
                y_score = np.append(seen_scores, unseen_scores)
                auc = roc_auc_score(y_true, y_score)
                print(f"    AUC: {auc}")
        
        if plotloss:
            plt.clf()
            #plt.plot(roc[0], roc[1])
            #plt.ylabel('TPR')
            #plt.xlabel('FPR')
            plt.plot(loss_arr_mean, 'b')
            plt.draw()
            plt.pause(0.001)
        epoch += 1

    seen_scores = get_membership_scores(N, test_seen_loc, nclasses,
            class_prototypes)
    unseen_scores = get_membership_scores(N, test_unseen_loc, nclasses,
            class_prototypes)
    y_true = np.append(np.ones(len(seen_scores)), np.zeros(len(unseen_scores)))
    y_score = np.append(seen_scores, unseen_scores)
    print(f'AUC: {roc_auc_score(y_true, y_score)}')
    if num_train_scores > 0:
        train_score_loc = np.random.choice(trainval_loc, num_train_scores,
                replace=False)
        train_score = get_membership_scores(N, train_score_loc, nclasses,
                class_prototypes)
        return y_score, y_true, train_score
    return y_score, y_true

if args.mode == 'val':
    print('\nCROSS VALIDATION\n')
    y_score_all = np.zeros(0)
    y_true_all = np.zeros(0)
    for i in range(3):
        y_score, y_true = train_eval(val_train_loc[i], val_test_seen_loc[i],
                val_test_unseen_loc[i])
        y_score_all = np.append(y_score_all, y_score)
        y_true_all = np.append(y_true_all, y_true)
    np.savetxt('y_true.txt', y_true_all, '%d')
    np.savetxt('y_score.txt', y_score_all)
'''
# Balance scores for each class by oversampling

def balance_class(y_score, y_true, loc):
    classes = np.unique(labels[loc])
    class_samples = np.bincount(labels[loc])
    max_samples = np.amax(class_samples) if class_samples.size > 0 else 0
    y_score_bal = np.zeros(0)
    y_true_bal = np.zeros(0, dtype=int)
    for c in classes:
        y_score_c = y_score[labels[loc] == c]
        y_true_c = y_true[labels[loc] == c]
        y_score_bal = np.append(y_score_bal, y_score_c)
        y_true_bal = np.append(y_true_bal, y_true_c)
        extra_idx = np.random.choice(np.arange(y_score_c.size), max_samples-class_samples[c])
        y_score_bal = np.append(y_score_bal, y_score_c[extra_idx])
        y_true_bal = np.append(y_true_bal, y_true_c[extra_idx])
    return y_score_bal, y_true_bal

y_score_bal_seen, y_true_bal_seen = balance_class(
    y_score_all[y_true_all.astype(bool)],
    y_true_all[y_true_all.astype(bool)],
    seen_loc_all
)

y_score_bal_unseen, y_true_bal_unseen = balance_class(
    y_score_all[~y_true_all.astype(bool)],
    y_true_all[~y_true_all.astype(bool)],
    unseen_loc_all
)

y_score_bal = np.append(y_score_bal_seen, y_score_bal_unseen)
y_true_bal = np.append(y_true_bal_seen, y_true_bal_unseen)
'''
if args.mode == 'test':
    print('\nTESTING\n')
    y_score_test, y_true_test, y_score_train = train_eval(trainval_loc, test_seen_loc,
             test_unseen_loc, num_train_scores=test_seen_loc.size)
    np.savetxt('y_score_test.txt', y_score_test)
    np.savetxt('y_score_train.txt', y_score_train)
#np.savetxt('train_seen_loc.txt', seen_loc_all, '%d')
#np.savetxt('train_unseen_loc.txt', unseen_loc_all, '%d')
#np.savetxt('y_true_bal.txt', y_true_bal, '%d')
#np.savetxt('y_score_bal.txt', y_score_bal)
