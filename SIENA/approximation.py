import edward as ed
import tensorflow as tf
import numpy as np
import math

class approximation:
	def __init__(self, model, data):
		self.model = model
		input_B = tf.layers.dense(model.X_data, 128, kernel_initializer=tf.truncated_normal_initializer(stddev=0.2))
		batch_normB = tf.layers.batch_normalization(input_B, axis=1, training=True, name="Hello1")
		hiddenB = tf.nn.relu(batch_normB)

		input_L = tf.layers.dense(model.X_data, 128, kernel_initializer=tf.truncated_normal_initializer(stddev=0.2))
		batch_normL = tf.layers.batch_normalization(input_L, axis=1, training=True, name="Hello2")
		hiddenL = tf.nn.relu(batch_normL)

		self.qBeta =  ed.models.TransformedDistribution(distribution=ed.models.Normal(
					loc = tf.layers.dense(hiddenB, model.n_genes), 
					scale = tf.layers.dense(hiddenB, model.n_genes, activation=tf.nn.softplus)),
  			bijector = tf.contrib.distributions.bijectors.Exp())

		self.qL =  ed.models.TransformedDistribution(distribution=ed.models.Normal(
					loc = tf.layers.dense(hiddenL, 1), 
					scale = tf.layers.dense(hiddenL, 1, activation=tf.nn.softplus)),
  			bijector = tf.contrib.distributions.bijectors.Exp())



	def define_inference(self, X, n_iter, n_samples, optimizer=None):
		optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=0.01)
		scale = {self.model.X: n_samples, self.model.Beta: n_samples, self.model.L: n_samples}
		if self.model.dispersion:
			self.vi = ed.ReparameterizationKLqp({self.model.Beta: self.qBeta, self.model.L: self.qL}, data={self.model.X: self.model.X_data})
			self.map = ed.MAP(data={self.model.X: self.model.X_data, self.model.Beta: self.qBeta, self.model.L: self.qL})
			self.map.initialize(n_iter=n_iter, var_list=[self.model.Theta], scale=scale)
		else:
			self.vi = ed.ReparameterizationKLqp({self.model.Beta: self.qBeta, self.model.L: self.qL}, data={self.model.X: self.model.X_data})
		train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		vi_vars = [v for v in train_vars if v.name != "Theta:0"]
		self.vi.initialize(var_list=vi_vars, n_iter=n_iter, optimizer=optimizer, scale=scale)



	def init_inference(self, sess):
		sess.run(tf.global_variables_initializer())



	def sample_infer(self, samples, samples_idx, sess=None):
		update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		self.vi.train = tf.group([self.vi.train, update_op])
		n_samples = len(samples)
		info_dict = {'loss': 0, 't': 0}
		for i in range(self.vi.n_iter):
			info_dict['loss'], info_dict['t'] = 0, i+1
			for n in range(n_samples):
				x_sample, sample_idx = samples[n], samples_idx[n]
				feed_dict = {self.model.X_data: x_sample}
				if self.model.X_idx is not None:
					feed_dict[self.model.X_idx] = sample_idx
				sample_dict = self.vi.update(feed_dict=feed_dict)
				if self.model.dispersion:
					sample_dict2 = self.map.update(feed_dict=feed_dict)
				if math.isnan(sample_dict['loss']):
					tf.add_check_numerics_ops().run(feed_dict=feed_dict)
				info_dict['loss'] += sample_dict['loss']
			info_dict['loss'] /= (n_samples*self.model.n_cells)
			self.vi.print_progress(info_dict)
		self.vi.finalize()
		if self.model.dispersion:
			self.map.finalize()
	
