from tensorflow.keras.utils import Sequence
from utils import generate_Xy
import math

class DatasetSequence(Sequence):
	def __init__(self, x_set, y_set, batch_size, labels, anchors, n_grid, net_input_size, n_class, normalize, aug):
		self.x, self.y = x_set, y_set
		self.batch_size = batch_size
		self.labels = labels
		self.anchors = anchors
		self.n_grid = n_grid
		self.net_input_size = net_input_size
		self.n_class = n_class
		self.normalize = normalize
		self.aug = aug

	def __len__(self):
		return math.ceil(len(self.x) / self.batch_size)

	def __getitem__(self, idx):
		batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
		# batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
		X_train, y_train = generate_Xy(batch_x, self.labels, self.anchors, self.n_grid, self.net_input_size, self.n_class, self.normalize, self.aug)
		return (X_train, y_train)