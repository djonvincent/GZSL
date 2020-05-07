Requirements:
  - scikit-learn
  - numpy
  - scipy
  - seaborn
  - torch
  - torchvision
  - matplotlib

These can be installed by running `pip install -r requirements.txt`.

You will need to download the image features from
http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip and copy the file
`res101.mat` from the relevant folder in data/ to the same folder here. E.g.
data/AWA2/res101.mat -> AWA2/res101.mat.

To get test results, run `./run_all_mnd.sh` followed by `./run_all_ale.sh`. The
test results will then appear as text files beginning with 'result-' in the
relevant dataset folder. E.g. the results for AWA2 with thresholding will be
saved as AWA2/result-threshold.txt. The results for ALE without novelty
detection are saved as 'result-baseline.txt'.

The first number is the seen accuracy, the second is the unseen accuracy, and
the third is their harmonic mean.
