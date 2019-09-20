import pandas as pd
import itertools 
import re
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from .model import *
from .approximation import *


class CSVDataset:
	"""counts_file,batches_file, labels_file must be csv files 
	- It extracts cell lables if labels_file is None - cells must be identified as <Label>_<Number>"""

	def __init__(self, folder,counts_file, gene_by_cell=True, batches_file=None, labels_file=None):
		X = pd.read_csv(os.path.join(folder, counts_file), index_col=0, header=0)
		if gene_by_cell:
			X = X.T
		if labels_file	is None:
			cell_names = X.index.values #column 0 is not a cell name
			labels = [re.sub(r'_[0-9]+', '', cell) for cell in cell_names]
			self.cell_types, self.labels = np.unique(labels,return_inverse=True)

		else:
			labels = pd.read_csv(os.path.join(folder, labels_file), header=0, index_col=0)
			self.cell_types, self.labels = np.unique(labels.values,return_inverse=True)

		self.batches, self.batch_idx = None,None
		if batches_file is not None:
			batches	 = pd.read_csv(os.path.join(folder, batches_file),header=0, index_col=0)
			self.batches, self.batch_idx = np.unique(batches.values,return_inverse=True)
		
		self.gene_names = np.array(X.columns, dtype=str)
		self.X = np.array(X.values)
		self.X = self.X.astype(np.float32)





class ModelDE():
	def __init__(self, X, types, labels, batches=None, batch_idx=None, zero_inflation=True, dispersion=True, use_log=True, minisample=None):
		ed.get_session().close() #close session and reset any previous model variables.
		tf.reset_default_graph()
		self.data = X
		self.types = list(types)
		self.labels_idx = labels
		self.batches = batches
		self.batch_idx = batch_idx
		self.sess = ed.get_session()
		self.lib_size = np.sum(self.data,axis=1) 

		self.model = Model(self.data.shape[0], self.data.shape[1], self.lib_size, dispersion=dispersion, use_log=use_log,
							minisample=minisample, batches=self.batches, batch_idx=self.batch_idx)
		if zero_inflation:
			p_dropout = self.ensemble_dropouts()
			self.model.set_ZI(p_dropout)

		self.samples, self.samples_idx, self.sample_sizes = self.model.generate_samples(X)
		


	def ensemble_dropouts(self):
		p_dic = {}
		probs = []
		if self.batches is None:
			for t in range(len(self.types)):
				counts = self.data[self.labels_idx == t]
				p_dic[t] = 1. - (np.count_nonzero(counts, axis=0)/counts.shape[0]) 																   
			for i in range(self.labels_idx.shape[0]):
				probs.append(p_dic[self.labels_idx[i]])

		else:
			for t in range(len(self.types)):
				for b in range(len(self.batches)):
					cells = np.logical_and(self.labels_idx == t, self.batch_idx == b)
					counts = self.data[cells]
					if counts.shape[0] == 0:
						continue
					p_dic[(t,b)] = 1. - (np.count_nonzero(counts, axis=0)/counts.shape[0]) 
			for i in range(self.labels_idx.shape[0]):
				probs.append(p_dic[(self.labels_idx[i],self.batch_idx[i])])

		print(p_dic)
		return np.array(probs)



	def approximate_model(self, optimizer=None, iters=1000):	
		print("Minisamples: " + str(len(self.samples)))
		self.approx = approximation(self.model, self.data)
		self.approx.define_inference(self.data, iters, len(self.samples), optimizer)
		self.approx.init_inference(self.sess)
		self.approx.sample_infer(self.samples, self.samples_idx, sess=self.sess)
	

	def eval_model(self):
		print("Loss: " + str(-self.sess.run(self.approx.vi.loss, feed_dict={self.model.X_data: self.data.astype('float32')}) / self.data.shape[0]))
		if self.model.dispersion:
			print(self.model.Theta.eval())
			x_post = ed.copy(self.model.X, {self.model.Beta: self.approx.qBeta, self.model.L: self.approx.qL})
		else:
			x_post = ed.copy(self.model.X, {self.model.Beta: self.approx.qBeta, self.model.L: self.approx.qL})
		print("Log-likelihood: " + str(ed.evaluate('log_likelihood', data={x_post: self.data.astype('float32'), self.model.X_data: self.data.astype('float32')})))



	def sample_rho_values(self, cells1, cells2, n_genes, n_samples=10): #n_samples -> number of mc samples
		n_t1, n_t2 = (np.where(cells1)[0]).shape[0], (np.where(cells2)[0]).shape[0]
		sample_op = self.approx.qBeta.sample(n_samples)
		model_n_samples = len(self.samples)
		all_rhos1 = np.array([np.zeros(shape=(n_t1, n_genes)) for _ in range(n_samples)])
		all_rhos2 = np.array([np.zeros(shape=(n_t2, n_genes)) for _ in range(n_samples)])
		i1, i2, rho1, rho2 = 0, 0, None, None
		for n in range(model_n_samples):
			x_sample, sample_idx, size = self.samples[n], self.samples_idx[n], self.sample_sizes[n]
			fetch1, fetch2 = cells1[sample_idx[0:size]], cells2[sample_idx[0:size]]
			sample_Beta = sample_op.eval(feed_dict={self.model.X_data: x_sample})
			for s in range(n_samples):
				sBeta = sample_Beta[s][0:size]
				rho1 = sBeta[fetch1,:] / np.sum(sBeta[fetch1,:], axis=1, keepdims=True)
				all_rhos1[s][i1: i1+rho1.shape[0],:] = rho1
				rho2 = sBeta[fetch2,:] / np.sum(sBeta[fetch2,:], axis=1, keepdims=True)
				all_rhos2[s][i2: i2+rho2.shape[0],:] = rho2
			i1 += rho1.shape[0]
			i2 += rho2.shape[0]
		return all_rhos1, all_rhos2



	def ensemble_pairs_by_bacthes(self, t1_cells, t2_cells, batches_names, batches,n_pairs=None):
		pairs, pairs_per_batch = [], []
		n_batches = batches_names.shape[0]
		b_t1, b_t2 = batches[t1_cells], batches[t2_cells]
		for b in range(0,n_batches):
			b_t1_ix, b_t2_ix = np.where(b_t1 == b)[0], np.where(b_t2 == b)[0]
			b_pairs = [b_t1_ix, b_t2_ix]
			pairs_per_batch.append(np.array(list(itertools.product(*b_pairs))))
		
		if n_pairs is not None:
			for b in range(0,n_batches):
				prop = np.count_nonzero(batches == b)/batches.shape[0]
				b_pairs = np.random.choice(np.arange(pairs_per_batch[b].shape[0]),size=round(n_pairs*prop), replace=False)
				pairs += pairs_per_batch[b][b_pairs,:].tolist()
		else:
			for b in range(0,n_batches):
				pairs += pairs_per_batch[b].tolist()
		return np.array(pairs)



	def ensemble_pairs(self, t1_cells, t2_cells, n_pairs=None):
		t1_ix, t2_ix = np.arange((np.where(t1_cells)[0]).shape[0]), np.arange((np.where(t2_cells)[0]).shape[0])
		pairs = [t1_ix, t2_ix]
		pairs = np.array(list(itertools.product(*pairs)))
		if n_pairs is not None:
			sub_pairs = np.random.choice(np.arange(pairs.shape[0]),size=n_pairs, replace=False)
			pairs = pairs[sub_pairs,:]
		return pairs



	def differential_expression_scores(self, type1, type2=None, n_samples=10, n_pairs=None):
		t1 = self.types.index(type1)
		cells_t1 = self.labels_idx == t1
		if type2 is not None:
			t2 = self.types.index(type2)
			cells_t2 = self.labels_idx == t2
		else:
			cells_t2 = self.labels_idx != t1

		if self.batches is None:
			pairs = self.ensemble_pairs(cells_t1, cells_t2, n_pairs)
		else:
			print("DE considering batches")
			pairs = self.ensemble_pairs_by_bacthes(cells_t1, cells_t2, self.batches, self.batch_idx, n_pairs)
		n_pairs = len(pairs)
		n_genes = self.data.shape[1]
		with tf.device("/cpu:0"):
			rho1_values, rho2_values = self.sample_rho_values(cells_t1, cells_t2, n_genes, n_samples)
		print("rho1_values shape: " + str(rho1_values.shape)) #3 dim (n_samples, n_cells, n_genes)
		print("rho2_values shape: " + str(rho2_values.shape))
		bayes_factors = np.zeros(shape=(n_pairs, n_genes))
		curr_pair = 0 
		cell1_rho, cell2_rho = np.zeros(shape=(n_samples,n_genes)), np.zeros(shape=(n_samples,n_genes))
		for pair in pairs:
			for i in range(n_samples):
				cell1_rho[i] = rho1_values[i,pair[0],:] #get rho samples for cell of type 1
				cell2_rho[i] = rho2_values[i,pair[1],:] #get rho samples for cell of type 2
			h1 = cell1_rho > cell2_rho # compare n_samples
			means = np.mean(h1, axis=0)# means.shape = (1,genes)
			pair_bayes = np.log(means + 1e-8) - np.log(1-means + 1e-8)
			bayes_factors[curr_pair] = pair_bayes
			curr_pair += 1

		means = np.mean(bayes_factors,axis=0) #bayes_factors matrix with shape (M_pairs, n_genes)
		return means



	def differentially_expressed_genes(self, gene_names, scores, threshold=0):
		abs_factors = np.abs(scores)
		df = pd.DataFrame(data={'Gene': gene_names,'factor': abs_factors})
		res = df.loc[df['factor'] >= threshold]
		res = res.sort_values(by=['factor'], ascending=False)
		return	res
