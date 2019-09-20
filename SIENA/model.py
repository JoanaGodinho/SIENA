import edward as ed
import tensorflow as tf
import numpy as np

class Model:
	def __init__(self, n_cells, n_genes, lib_size, dispersion=True, use_log=True, minisample=None, batches=None, batch_idx=None):
		self.n_cells = n_cells
		self.n_genes = n_genes
		self.dispersion = dispersion
		self.lib_size = lib_size
		self.minisample = n_cells if minisample is None else minisample
		log_lib = np.log(lib_size) if use_log else None

		self.X_data = tf.placeholder(tf.float32, [self.minisample, n_genes])
		if batches is None:
			self.X_idx = None 
			a1 = np.mean(log_lib) if use_log else np.square(np.mean(lib_size))/np.square(np.std(lib_size)) 
			a2 = np.std(log_lib) if use_log else np.mean(lib_size)/np.square(np.std(lib_size)) 
		else:
			a1, a2 = np.zeros(shape=(n_cells, 1), dtype=np.float32), np.zeros(shape=(n_cells, 1), dtype=np.float32)
			for b in range(batches.shape[0]):
				b_sizes =  log_lib[batch_idx == b] if use_log else lib_size[batch_idx == b]
				a1[batch_idx == b] =  np.mean(b_sizes) if use_log else np.square(np.mean(b_sizes))/np.square(np.std(b_sizes)) 
				a2[batch_idx == b] =  np.std(b_sizes) if use_log else np.mean(b_sizes)/np.square(np.std(b_sizes)) 
			self.X_idx = tf.placeholder(tf.int32, [self.minisample])
			a1 = tf.gather(a1, self.X_idx)
			a2 = tf.gather(a2, self.X_idx)

		self.Beta = ed.models.Gamma(concentration=(1/3)*tf.ones([self.minisample,n_genes]), rate=1*tf.ones([self.minisample,n_genes]))
			# Rho = Beta/ Sum(Beta)
		self.rho = self.Beta/tf.reduce_sum(self.Beta, 1, keepdims=True)
			# L ~ Log-Normal(a1,a2) or ~ Gamma(a1,a2)
		if use_log:
			print("Using log scalings")
			self.L = ed.models.TransformedDistribution(distribution=ed.models.Normal(
					loc=a1*tf.ones([self.minisample,1]), scale= a2*tf.ones([self.minisample,1])),
	  			bijector=tf.contrib.distributions.bijectors.Exp())
		else:
			self.L = ed.models.Gamma(concentration=a1*tf.ones([self.minisample,1]), rate=a2*tf.ones([self.minisample,1]))
		
		if self.dispersion:
			self.Theta = tf.Variable(tf.zeros([1,n_genes]), name="Theta")#ed.models.Gamma(concentration=tf.ones([1,n_genes]), rate=1*tf.ones([1,n_genes]))
			Theta = tf.nn.softplus(self.Theta)
		else:
			print("No Dispersion")
			Theta = tf.ones([1,n_genes])

		r = tf.tile(self.L,[1, n_genes])* tf.tile(Theta,[self.minisample,1]) *self.rho
		p = tf.ones([self.minisample,n_genes])/(tf.ones([self.minisample,n_genes])+tf.tile(Theta,[self.minisample,1]))
		self.nb = ed.models.NegativeBinomial(total_count=r, probs=p)
		self.X = self.nb



	def set_ZI(self, p):
		print("Using Dropout")
		p_dropout = p[:, :, np.newaxis]
		self.mixture_probs = tf.constant(np.dstack((p_dropout, 1.- p_dropout))) #matrix (n_cells, n_genes, 2)
		if self.X_idx is None:
			self.X_idx = tf.placeholder(tf.int32, [self.minisample])
		probs = tf.gather(self.mixture_probs, self.X_idx)
		catg = ed.models.Categorical(probs=tf.cast(probs,tf.float32))
		# X ~ P_dropout*Deterministic(0) + (1-P_dropout)*NB() -- NB = Poisson(lambda)
		self.X = ed.models.Mixture(cat=catg,components=[ed.models.Deterministic(loc=tf.zeros([self.minisample,self.n_genes])), self.nb])



	def generate_samples(self, X):
		samples, samples_idx, sizes = [], [], []
		sample_size = self.minisample
		for i in range(0, self.n_cells, self.minisample):
			if i + self.minisample > self.n_cells:
				a = np.arange(i, self.n_cells)
				b = np.random.choice(np.arange(self.n_cells), size = self.minisample - len(a), replace=False)
				sample_idx = np.concatenate((a, b), axis=None)
				sample_size = len(a)
			else:
				sample_idx = np.arange(i, i + self.minisample)
			samples.append(X[sample_idx,:])
			samples_idx.append(sample_idx)
			sizes.append(sample_size)
		return samples, samples_idx, sizes


	def random_sample(self, X):
		sample_idx = np.random.choice(self.n_cells, size=self.minisample, replace=False)
		return X[sample_idx,:], sample_idx