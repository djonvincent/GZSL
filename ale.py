import torch
import torch.nn.functional as F
import numpy as np
import scipy.io
import argparse
import os
from model import Compatibility
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--mode', choices=['test','val'], type=str, default='val')
parser.add_argument('--novelty', action='store_true', dest='novelty')
parser.add_argument('--confusion', action='store_true', dest='confusion')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--wd', type=float, default=0.0001)
parser.add_argument('--b', type=float, default=1)
parser.add_argument('--result-file', type=str, default=None)
parser.set_defaults(novelty=False, confusion=False)
args = parser.parse_args()
os.chdir(args.dataset)

def class_acc(y_, y):
    cls = torch.unique(y)
    total_acc = 0
    for c in cls:
        idx = y == c
        acc = torch.sum(y_[idx] == c).item()/torch.sum(idx).item()
        total_acc += acc
    total_acc /= cls.size(0)
    return total_acc

def load_loc(fn):
    loc = []
    with open(fn) as f:
        for l in f.readlines():
            loc.append(np.array([int(x) for x in l.split(' ')], dtype=int))
    return loc

def index_labels(classes, labels):
    labels2 = np.searchsorted(classes, labels)
    return labels2

batch_size = args.batch_size
device = torch.device('cuda')

res101 = scipy.io.loadmat('res101.mat')
splits = scipy.io.loadmat('att_splits.mat')
features = res101['features'].transpose()
att = splits['att'].transpose()
labels = res101['labels'][:,0]-1

if args.mode == 'test':
    train_loc = splits['trainval_loc'][:,0]-1
    test_seen_loc = splits['test_seen_loc'][:,0]-1
    test_unseen_loc = splits['test_unseen_loc'][:,0]-1
    trials = 1
    if args.novelty:
        nov_s = torch.from_numpy(
            np.loadtxt('novelty_seen.txt')
        ).float().view(-1,1).to(device)
        nov_u = torch.from_numpy(
            np.loadtxt('novelty_unseen.txt')
        ).float().view(-1,1).to(device)
    print('TESTING')

elif args.mode == 'val':
    trials = 3
    print('VALIDATION')

all_acc_zsl = np.zeros(trials)
all_acc_u = np.zeros(trials)
all_acc_s = np.zeros(trials)
all_h_score = np.zeros(trials)
all_bias = np.zeros(trials)
for trial in range(trials):
    if args.mode == 'val':
        train_loc = load_loc('val_train_loc.txt')[trial]
        test_seen_loc = load_loc('val_test_seen_loc.txt')[trial]
        test_unseen_loc = load_loc('val_test_unseen_loc.txt')[trial]
        print(f'SPLIT {trial+1}')

    s_classes = np.unique(labels[train_loc])
    u_classes = np.unique(labels[test_unseen_loc])
    print(f'# Seen classes: {s_classes.size}')
    print(f'# Unseen classes: {u_classes.size}')
    print(f'# Train samples: {train_loc.size}')
    print(f'# Test samples: {test_seen_loc.size + test_unseen_loc.size}')
    classes = np.sort(np.append(s_classes, u_classes))
    attr = torch.from_numpy(att[classes]).float().to(device)
    s_attr = torch.from_numpy(att[s_classes]).float().to(device)
    u_attr = torch.from_numpy(att[u_classes]).float().to(device)
    x_train = torch.from_numpy(features[train_loc]).float().to(device)
    x_s_test = torch.from_numpy(features[test_seen_loc]).float().to(device)
    x_u_test = torch.from_numpy(features[test_unseen_loc]).float().to(device)
    y_train_ix = torch.from_numpy(
        index_labels(s_classes, labels[train_loc])
    ).long().to(device)
    y_s_test = torch.from_numpy(
        index_labels(classes, labels[test_seen_loc])
    ).long().to(device)
    y_s_test_ix = torch.from_numpy(
        index_labels(s_classes, labels[test_seen_loc])
    ).long().to(device)
    y_u_test = torch.from_numpy(
        index_labels(classes, labels[test_unseen_loc])
    ).long().to(device)
    y_u_test_ix = torch.from_numpy(
        index_labels(u_classes, labels[test_unseen_loc])
    ).long().to(device)

    mask = torch.zeros(len(classes)).bool()
    for c in range(len(classes)):
        mask[c] = classes[c] in u_classes

    np.random.seed(123)
    torch.manual_seed(123)

    data = TensorDataset(x_train, y_train_ix)
    dataloader = DataLoader(data, batch_size, shuffle=True, drop_last=False)

    N = Compatibility(features.shape[1], att.shape[1]).to(device)
    optim = torch.optim.Adam(params=N.parameters(), lr=args.lr, weight_decay=args.wd)

    epoch = 0
    ce = torch.nn.CrossEntropyLoss()
    loss_arr = np.zeros(0)
    while epoch < args.epochs:
        print(f'Epoch {epoch+1}')
        N.train()
        epoch_loss = 0
        for x, y in dataloader:
            y_ = N(x,s_attr)
            loss = ce(y_,y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_arr = np.append(loss_arr, loss.item())
            epoch_loss += loss.item() * batch_size
        print(f'    Mean loss: {loss_arr.mean()}')
        print(f'    Epoch loss: {running_loss/train_loc.size}')
        N.eval()
        y_ = N(x_u_test, u_attr)
        y_ = torch.argmax(y_, dim=1)
        acc_zsl = class_acc(y_, y_u_test_ix)
        ys_ = N(x_s_test, attr)
        if args.novelty:
            ys_[:, mask] *= nov_s
            ys_[:, ~mask] *= (1-nov_s)
        ys_ = torch.argmax(ys_, dim=1)
        acc_s = class_acc(ys_, y_s_test)
        yu_ = N(x_u_test, attr)
        if args.novelty:
            yu_[:, mask] *= nov_u
            yu_[:, ~mask] *= (1-nov_u)
        yu_ = torch.argmax(yu_, dim=1)
        acc_u = class_acc(yu_, y_u_test)
        yuu_ = N(x_u_test, u_attr).max(dim=1).values
        yus_ = N(x_u_test, s_attr).max(dim=1).values
        yss_ = N(x_s_test, s_attr).max(dim=1).values
        ysu_ = N(x_s_test, u_attr).max(dim=1).values
        m1 = (yuu_ > 0) & (yus_ > 0)
        m2 = (yss_ > 0) & (ysu_ > 0)
        bias = ((yus_[m1]/yuu_[m1]).mean() + (yss_[m2]/ysu_[m2]).mean())/2
        h_score = 2*acc_u*acc_s/(acc_u+acc_s)
        print(f'    Bias: {bias}')
        print(f'    Acc. ZSL: {acc_zsl}')
        print(f'    Acc. seen: {acc_s}')
        print(f'    Acc. unseen: {acc_u}')
        print(f'    H-score: {h_score}')
        epoch += 1
    if args.confusion:
        mapping = np.zeros(classes.size)
        mapping[index_labels(classes, s_classes)] = np.arange(0,s_classes.size,1)
        mapping[index_labels(classes, u_classes)] = np.arange(s_classes.size, classes.size, 1)
        x_conf = torch.cat((ys_, yu_)).cpu().detach().numpy()
        x_conf = mapping[x_conf]
        y_conf = torch.cat((y_s_test, y_u_test)).cpu().detach().numpy()
        y_conf = mapping[y_conf]
        conf_matrix = confusion_matrix(
            x_conf,
            y_conf,
            normalize='pred'
        )
        sns.heatmap(conf_matrix)
        plt.show()
    all_acc_zsl[trial] = acc_zsl
    all_acc_u[trial] = acc_u
    all_acc_s[trial] = acc_s
    all_h_score[trial] = h_score
    all_bias[trial] = bias

    yuu_ = N(x_u_test, u_attr).max(dim=1)
    yus_ = N(x_u_test, s_attr).max(dim=1)
    yss_ = N(x_s_test, s_attr).max(dim=1)
    ysu_ = N(x_s_test, u_attr).max(dim=1)
    uix = yuu_.indices == y_u_test_ix
    six = yss_.indices == y_s_test_ix
    yuu_ = yuu_.values
    yus_ = yus_.values
    yss_ = yss_.values
    ysu_ = ysu_.values
    np.savetxt('ix.txt', torch.cat((uix, six)).cpu().detach().numpy(), '%d')
    np.savetxt('cu.txt',torch.cat((yuu_, ysu_)).cpu().detach().numpy())
    np.savetxt('cs.txt', torch.cat((yus_, yss_)).cpu().detach().numpy())
    np.savetxt('y.txt', np.append(-np.ones(yuu_.size(0)), np.ones(ysu_.size(0))))
print(f'Bias: {all_bias.mean()}')
print(f'Acc. ZSL: {all_acc_zsl.mean()}')
print(f'Acc. seen: {all_acc_s.mean()}')
print(f'Acc. unseen: {all_acc_u.mean()}')
print(f'H-score: {all_h_score.mean()}')
if args.result_file:
    with open(args.result_file, 'w') as f:
        f.write(f'{all_acc_s.mean()} {all_acc_u.mean()} {all_h_score.mean()}')
