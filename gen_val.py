import numpy as np
import argparse
import scipy.io
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()
os.chdir(args.dataset)

res101 = scipy.io.loadmat('res101.mat')
splits = scipy.io.loadmat('att_splits.mat')
labels = res101['labels'][:,0]
trainval_loc = splits['trainval_loc'][:,0]-1
test_seen_loc = splits['test_seen_loc'][:,0]-1
test_unseen_loc = splits['test_unseen_loc'][:,0]-1
n_seen_classes = len(np.unique(labels[trainval_loc]))
n_unseen_classes = len(np.unique(labels[test_unseen_loc]))
split_ratio = n_seen_classes/(n_seen_classes + n_unseen_classes)

val_train_loc = []
val_test_seen_loc = []
val_test_unseen_loc = []

def write(arr, fname):
    with open(fname, 'w') as f:
        f.write('\n'.join([' '.join([str(int(x)) for x in l]) for l in arr]))

for i in range(3):
    seen_classes = np.random.choice(
        np.unique(labels[trainval_loc]),
        int(split_ratio*n_seen_classes),
        replace=False
    )
    seen_loc = trainval_loc[np.isin(labels[trainval_loc], seen_classes)]
    np.random.shuffle(seen_loc)
    # Hold back 20% of seen data for testing
    test_unseen_loc_subset = trainval_loc[~np.isin(labels[trainval_loc], seen_classes)]
    test_seen_loc_subset, train_loc_subset = np.split(seen_loc, [seen_loc.size//5])
    val_train_loc.append(train_loc_subset)
    val_test_seen_loc.append(test_seen_loc_subset)
    val_test_unseen_loc.append(test_unseen_loc_subset)

write(val_train_loc, 'val_train_loc.txt')
write(val_test_seen_loc, 'val_test_seen_loc.txt')
write(val_test_unseen_loc, 'val_test_unseen_loc.txt')
