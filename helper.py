from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from os import listdir
from os.path import join
import numpy as np

def calculate_auc(scrs, lbls, trueDEG, sym=False):
	scores = np.array(scrs)
	if sym:
		scores = 1 - scores
	labels = np.array(lbls.isin(trueDEG))
	fpr,tpr,thresh = roc_curve(labels, scores)
	return auc(fpr, tpr)


def average_scores(dirr):
	files = listdir(dirr)
	for i in range(len(files)):
		path = join(dirr, files[i])
		res = pd.read_csv(path, sep=",", header=0, names=["Gene", "factor"+str(i)])
		if i == 0:
			agg = res
		else:
			agg = pd.merge(agg, res, on="Gene")
	agg['median'] = agg.median(axis=1)
	return agg