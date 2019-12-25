from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, Reshape, MaxPooling2D, BatchNormalization, LeakyReLU, concatenate
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import MobileNet

from classification_models.tfkeras import Classifiers

import tensorflow as tf
import numpy as np
import cv2

from utils import load_annotation, compute_overlap, compute_ap, decode_netout, generate_Xy

class BaseNet(object):
	"""docstring for BaseNet"""
	def __init__(self, net_input_size, anchors, n_class, weights_dir, labels):
		self.net_input_size = net_input_size
		self.anchors = anchors
		self.n_bounding_box = len(anchors) // 2
		self.n_class = n_class
		self.weights_dir = weights_dir
		self.labels = labels
		
	def create_model(self, feature_extractor):
		self.n_grid = feature_extractor.get_output_shape_at(-1)[1:3][0]

		input_layer = Input(shape=(self.net_input_size, self.net_input_size, 3))

		feature = feature_extractor(input_layer)

		# Detection Layer
		output = Conv2D(self.n_bounding_box * (4 + 1 + self.n_class), 
								(1,1), strides=(1,1), 
								padding='same', 
								name='DetectionLayer', 
								kernel_initializer='lecun_normal')(feature)
		output = Reshape((self.n_grid, self.n_grid, self.n_bounding_box, 4 + 1 + self.n_class))(output)
		
		final_yolo_model = Model(input_layer, output)
		layer = final_yolo_model.layers[-2]
		
		weights = layer.get_weights()

		new_kernel = np.random.normal(size=weights[0].shape)/(self.n_grid*self.n_grid)
		new_bias   = np.random.normal(size=weights[1].shape)/(self.n_grid*self.n_grid)

		layer.set_weights([new_kernel, new_bias])
		
		return final_yolo_model

	def set_generator(self, imgs, batch_size, aug):
		pointer = 0
		# in_loop = False
		while True:
			# if in_loop:
			# 	in_loop = True
			# 	del X_train
			# 	del y_train
			imgs_generated = []
			for i in range(batch_size):
				if pointer == len(imgs):
					pointer = 0
				imgs_generated.append(imgs[pointer])
				pointer += 1
			X_train, y_train = generate_Xy(imgs_generated, self.labels, self.anchors, self.n_grid, self.net_input_size, self.n_class, self.normalize, aug=aug)
			yield (X_train, y_train)

	def custom_loss(self, y_true, y_pred):
		anchors = self.anchors
		n_bounding_box = self.n_bounding_box
		row, col = self.n_grid, self.n_grid
		batch_size = tf.shape(y_true)[0]
		n_class = self.n_class

		class_wt = np.ones(n_class, dtype='float32')

		no_object_scale = 1
		object_scale = 1
		class_scale = 1
		coord_scale = 1.5

		# (#, 7, 7, 5)
		mask_shape = tf.shape(y_true)[:4]
		# (1, 7, 7, 1, 1)
		cell_x = tf.cast(tf.reshape(tf.tile(tf.range(row), [col]), (1, row, col, 1, 1)), dtype=tf.float32)
		# (1, 7, 7, 1, 1)
		cell_y = tf.transpose(cell_x, (0,2,1,3,4))
		# (32, 7, 7, 5, 2)
		cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [batch_size, 1, 1, n_bounding_box, 1])

		coord_mask = tf.zeros(mask_shape)
		conf_mask  = tf.zeros(mask_shape)
		class_mask = tf.zeros(mask_shape)

		seen = tf.Variable(0.)
		total_recall = tf.Variable(0.)

		"""
		Adjust prediction
		"""
		### adjust x and y      
		# (32, 7, 7, 5, 2)
		pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

		### adjust w and h
		# (32, 7, 7, 5, 2)
		pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(anchors, [1,1,1,n_bounding_box,2])

		### adjust confidence
		# (32, 7, 7, 5, 1*)
		pred_box_conf = tf.sigmoid(y_pred[..., 4])

		### adjust class probabilities
		# (32, 7, 7, 5, 2) <- # of class
		pred_box_class = y_pred[..., 5:]

		"""
		Adjust ground truth
		"""
		### adjust x and y
		true_box_xy = y_true[..., 0:2] # relative position to the containing cell

		### adjust w and h
		true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically

		### adjust confidence

		# max and min x and y
		true_wh_half = true_box_wh / 2.
		true_mins    = true_box_xy - true_wh_half
		true_maxes   = true_box_xy + true_wh_half

		pred_wh_half = pred_box_wh / 2.
		pred_mins    = pred_box_xy - pred_wh_half
		pred_maxes   = pred_box_xy + pred_wh_half       

		# calculate IoU
		intersect_mins  = tf.maximum(pred_mins,  true_mins)
		intersect_maxes = tf.minimum(pred_maxes, true_maxes)
		intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
		intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

		true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
		pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

		union_areas = pred_areas + true_areas - intersect_areas
		iou_scores  = tf.truediv(intersect_areas, union_areas)

		true_box_conf = iou_scores * y_true[..., 4]

		### adjust class probabilities
		# (32, 7, 7, 5, 1*)
		true_box_class = tf.argmax(y_true[..., 5:], -1)

		"""
		Determine the masks
		"""

		### coordinate mask: simply the position of the ground truth boxes (the predictors)
		coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * coord_scale

		### confidence mask: penelize predictors + penalize boxes with low IOU
		iou_scores = tf.expand_dims(iou_scores, axis=-1)
		best_ious = tf.reduce_max(iou_scores, axis=-1)
		conf_mask = conf_mask + tf.cast(best_ious < 0.6, dtype=tf.float32) * (1 - y_true[..., 4]) * no_object_scale

		#     print(conf_mask)

		# penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
		conf_mask = conf_mask + y_true[..., 4] * object_scale

		### class mask: simply the position of the ground truth boxes (the predictors)
		class_mask = y_true[..., 4] * tf.gather(class_wt, true_box_class) * class_scale       

		"""
		Finalize the loss
		"""
		nb_coord_box = tf.reduce_sum(tf.cast(coord_mask > 0.0, dtype=tf.float32))
		nb_conf_box  = tf.reduce_sum(tf.cast(conf_mask  > 0.0, dtype=tf.float32))
		n_class_box = tf.reduce_sum(tf.cast(class_mask > 0.0, dtype=tf.float32))

		loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)* coord_mask) / (nb_coord_box + 1e-6) / 2.
		loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)* coord_mask) / (nb_coord_box + 1e-6) / 2.

		loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf)* conf_mask)  / (nb_conf_box  + 1e-6) / 2.
		loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
		loss_class = tf.reduce_sum(loss_class * class_mask) / (n_class_box + 1e-6)


		loss = loss_xy + loss_wh + loss_conf + loss_class

		return loss

	def compile_model(self, final_yolo_model, optimizer):

		final_yolo_model.compile(loss=self.custom_loss, optimizer=optimizer)
		return final_yolo_model

	def predict(self, model, image, obj_threshold, nms_threshold):
		'''
		# Arguments
			model          : Model.
			image          : Raw image.
		# Returns
			Bounding boxes predicted.
		'''

		image_h, image_w, _ = image.shape
		image = cv2.resize(image, (self.net_input_size, self.net_input_size))
		image = self.normalize(image)

		input_image = image[:,:,::-1]
		input_image = np.expand_dims(input_image, 0)

		netout = model.predict(input_image)[0]
		boxes  = decode_netout(netout, self.anchors, self.n_class, obj_threshold=obj_threshold, nms_threshold=nms_threshold)

		return boxes

	def evaluate(self, model, imgs, obj_threshold=0.3, nms_threshold=0.3, iou_threshold=0.5):
		"""
		# Arguments
			model           : The model to evaluate.
			imgs            : list of parsed test_img dictionaries.
			obj_threshold 	: The score confidence threshold to use for detections.
			nms_threshold   : The threshold used to consider when a detection is positive or negative.
		# Returns
			A dict mapping class names to mAP scores.
		"""    
		# gather all detections and annotations

		test_size = len(imgs)

		all_detections     = [[None for i in range(self.n_class)] for j in range(test_size)]
		all_annotations    = [[None for i in range(self.n_class)] for j in range(test_size)]
		ious = []

		for i in range(test_size):

			image_name = imgs[i]['filename']
			
			if '.jpg' not in image_name and '.png' not in image_name:
				image_name += '.jpg'

			raw_image = cv2.imread(image_name)

			raw_height, raw_width, raw_channels = raw_image.shape

			# make the boxes and the labels
			pred_boxes  = self.predict(model, raw_image, obj_threshold, nms_threshold)
			
			score = np.array([box.score for box in pred_boxes])
			pred_labels = np.array([box.label for box in pred_boxes])        
			
			if len(pred_boxes) > 0:
				pred_boxes = np.array([[box.xmin*raw_width, 
					box.ymin*raw_height, 
					box.xmax*raw_width, 
					box.ymax*raw_height, 
					box.score] for box in pred_boxes])
			else:
				pred_boxes = np.array([[]])  
			
			# sort the boxes and the labels according to scores
			score_sort = np.argsort(-score)
			pred_labels = pred_labels[score_sort]
			pred_boxes  = pred_boxes[score_sort]
			
			# copy detections to all_detections
			for label in range(self.n_class):
				all_detections[i][label] = pred_boxes[pred_labels == label, :]
			
			annotations = load_annotation(imgs, i, self.labels)
			
			# copy detections to all_annotations
			for label in range(self.n_class):
				all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
				
		# compute mAP by comparing all detections and all annotations
		average_precisions = {}
		
		for label in range(self.n_class):
			false_positives = np.zeros((0,))
			true_positives  = np.zeros((0,))
			scores          = np.zeros((0,))
			num_annotations = 0.0

			for i in range(test_size):
				detections           = all_detections[i][label]
				annotations          = all_annotations[i][label]
				num_annotations     += annotations.shape[0]
				detected_annotations = []

				for d in detections:
					scores = np.append(scores, d[4])

					if annotations.shape[0] == 0:
						false_positives = np.append(false_positives, 1)
						true_positives  = np.append(true_positives, 0)
						continue

					overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
					assigned_annotation = np.argmax(overlaps, axis=1)
					max_overlap         = overlaps[0, assigned_annotation]

					ious.append(max_overlap)

					if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
						false_positives = np.append(false_positives, 0)
						true_positives  = np.append(true_positives, 1)
						detected_annotations.append(assigned_annotation)
					else:
						false_positives = np.append(false_positives, 1)
						true_positives  = np.append(true_positives, 0)

			# no annotations -> AP for this class is 0 (is this correct?)
			if num_annotations == 0:
				average_precisions[label] = 0
				continue

			# sort by score
			indices         = np.argsort(-scores)
			false_positives = false_positives[indices]
			true_positives  = true_positives[indices]

			# compute false positives and true positives
			false_positives = np.cumsum(false_positives)
			true_positives  = np.cumsum(true_positives)

			# compute recall and precision
			recall    = true_positives / num_annotations
			precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

			# compute average precision
			average_precision  = compute_ap(recall, precision)  
			average_precisions[label] = average_precision

		map_dict = {}
		# print evaluation
		for label, average_precision in average_precisions.items():
			map_dict[self.labels[label]] = average_precision
			print(self.labels[label], '{:.4f}'.format(average_precision))
		
		print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))
		print('average IOU: {:.4f}'.format(np.mean(ious)))
		
		average_map = sum(average_precisions.values()) / len(average_precisions)

		return [average_map, map_dict, np.mean(ious)]
		# return average_precisions    

	def predict_boxes(self, model, image, obj_threshold, nms_threshold):
		boxes = self.predict(model, image, obj_threshold=obj_threshold, nms_threshold=nms_threshold)
		image_h, image_w, _ = image.shape
		count_dict = {0: 0, 1: 0}
		prediction_info = {'coords': {0:[], 1:[]}, 'total_count': len(boxes)}

		# with open('coords.txt', 'a+') as f:
		for box in boxes:
			x_center, y_center = box.get_center()
			x_center = int(x_center*image_w)
			y_center = int(y_center*image_h)
			# prediction_info['coords'].append((x_center, y_center))
			
			# f.write(""+str(x_center)+ ' ' + str(y_center) + '\n')

			xmin = int(box.xmin*image_w)
			ymin = int(box.ymin*image_h)
			xmax = int(box.xmax*image_w)
			ymax = int(box.ymax*image_h)

			if box.get_label() == 0:
				cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,0,255), 3)
				prediction_info['coords'][0].append((x_center, y_center))

				if 0 in count_dict:
					count_dict[0] += 1
				# else:
				# 	count_dict[0] = 1
				# cv2.putText(image, 
				# 			self.labels[box.get_label()] + ' ' + str(box.get_score()), 
				# 			(xmin, ymin - 13), 
				# 			cv2.FONT_HERSHEY_SIMPLEX, 
				# 			1e-3 * image_h, 
				# 			(0,255,0), 2)
			else:
				cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 3)
				prediction_info['coords'][1].append((x_center, y_center))
				if 1 in count_dict:
					count_dict[1] += 1
				# else:
				# 	count_dict[1] = 1
				# cv2.putText(image, 
				# 			self.labels[box.get_label()] + ' ' + str(box.get_score()), 
				# 			(xmin, ymin - 13), 
				# 			cv2.FONT_HERSHEY_SIMPLEX, 
				# 			1e-3 * image_h, 
				# 			(0,255,0), 2)
		# f.write("---\n")
		# ---- <Indent> ----
		# with open('counts.txt', 'a+') as c:
		# 	c.write(str(len(boxes)))
		# 	c.write('\n')
		cv2.putText(image, 'bee-A: ' + str(count_dict[0]), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[1], (0,255,0), 2)
		if len(self.labels) > 1:
				cv2.putText(image, 'bee-B: ' + str(count_dict[1]), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[1], (0,255,0), 2)
		prediction_info['count_dict'] = count_dict
		return prediction_info

	def normalize(self, image):
		raise NotImplementedError("Not implemented yet.")       

class VGG16Net(BaseNet):
	"""docstring for VGG16"""
	def __init__(self, net_input_size, anchors, n_class, weights_dir, labels):
		super(VGG16Net, self).__init__(net_input_size, anchors, n_class, weights_dir, labels)

	def create_base_network(self, transfer_learning=False):
		model = VGG16(include_top=False, 
			weights=self.weights_dir, 
			input_shape=(self.net_input_size, self.net_input_size, 3), 
			pooling='avg')
		
		feature_extractor = Model(model.layers[0].input, model.layers[-2].output) 

		if transfer_learning:
			for l in feature_extractor.layers:
				l.trainable = False

		# feature_extractor.summary()
		return feature_extractor

	def normalize(self, reshaped_image):
		def norm(reshaped_image):
			reshaped_image = reshaped_image.astype('float')
			reshaped_image[..., 0] -= 103.939
			reshaped_image[..., 1] -= 116.779
			reshaped_image[..., 2] -= 123.68
			return reshaped_image
		return norm(reshaped_image)

class TinyYoloNet(BaseNet):
	"""docstring for Tiny YOLO Net"""
	def __init__(self, net_input_size, anchors, n_class, weights_dir, labels):
		super(TinyYoloNet, self).__init__(net_input_size, anchors, n_class, weights_dir, labels)

	def create_base_network(self, transfer_learning=False):
		input_image = Input(shape=(self.net_input_size, self.net_input_size, 3))
		# Layer 1
		x = Conv2D(16, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
		x = BatchNormalization(name='norm_1')(x)
		x = LeakyReLU(alpha=0.1)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

		# Layer 2 - 5
		for i in range(0,4):
			x = Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', name='conv_' + str(i+2), use_bias=False)(x)
			x = BatchNormalization(name='norm_' + str(i+2))(x)
			x = LeakyReLU(alpha=0.1)(x)
			x = MaxPooling2D(pool_size=(2, 2))(x)

		# Layer 6
		x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
		x = BatchNormalization(name='norm_6')(x)
		x = LeakyReLU(alpha=0.1)(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

		# Layer 7 - 8
		for i in range(0,2):
			x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_' + str(i+7), use_bias=False)(x)
			x = BatchNormalization(name='norm_' + str(i+7))(x)
			x = LeakyReLU(alpha=0.1)(x)

		feature_extractor = Model(input_image, x) 
		feature_extractor.load_weights(self.weights_dir)

		if transfer_learning:
			for l in feature_extractor.layers:
				l.trainable = False

		# feature_extractor.summary()
		return feature_extractor

	def normalize(self, reshaped_image):
		def norm(reshaped_image):
			reshaped_image = reshaped_image.astype('float')
			reshaped_image /= 255.
			return reshaped_image
		return norm(reshaped_image)

class YoloNet(BaseNet):
	"""docstring for Full YOLO Net"""
	def __init__(self, net_input_size, anchors, n_class, weights_dir, labels):
		super(YoloNet, self).__init__(net_input_size, anchors, n_class, weights_dir, labels)

	def create_base_network(self, transfer_learning=False):
		# Layer 1
		input_image = Input(shape=(self.net_input_size, self.net_input_size, 3))

		# the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
		def space_to_depth_x2(x):
			return tf.nn.space_to_depth(x, block_size=2)

		# Layer 1
		x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
		x = BatchNormalization(name='norm_1')(x)
		x = LeakyReLU(alpha=0.1)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

		# Layer 2
		x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
		x = BatchNormalization(name='norm_2')(x)
		x = LeakyReLU(alpha=0.1)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

		# Layer 3
		x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
		x = BatchNormalization(name='norm_3')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 4
		x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
		x = BatchNormalization(name='norm_4')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 5
		x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
		x = BatchNormalization(name='norm_5')(x)
		x = LeakyReLU(alpha=0.1)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

		# Layer 6
		x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
		x = BatchNormalization(name='norm_6')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 7
		x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
		x = BatchNormalization(name='norm_7')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 8
		x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
		x = BatchNormalization(name='norm_8')(x)
		x = LeakyReLU(alpha=0.1)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

		# Layer 9
		x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
		x = BatchNormalization(name='norm_9')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 10
		x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
		x = BatchNormalization(name='norm_10')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 11
		x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
		x = BatchNormalization(name='norm_11')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 12
		x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
		x = BatchNormalization(name='norm_12')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 13
		x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
		x = BatchNormalization(name='norm_13')(x)
		x = LeakyReLU(alpha=0.1)(x)

		skip_connection = x

		x = MaxPooling2D(pool_size=(2, 2))(x)

		# Layer 14
		x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
		x = BatchNormalization(name='norm_14')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 15
		x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
		x = BatchNormalization(name='norm_15')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 16
		x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
		x = BatchNormalization(name='norm_16')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 17
		x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
		x = BatchNormalization(name='norm_17')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 18
		x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
		x = BatchNormalization(name='norm_18')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 19
		x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
		x = BatchNormalization(name='norm_19')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 20
		x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
		x = BatchNormalization(name='norm_20')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 21
		skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
		skip_connection = BatchNormalization(name='norm_21')(skip_connection)
		skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
		skip_connection = Lambda(space_to_depth_x2)(skip_connection)

		x = concatenate([skip_connection, x])

		# Layer 22
		x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
		x = BatchNormalization(name='norm_22')(x)
		x = LeakyReLU(alpha=0.1)(x)

		feature_extractor = Model(input_image, x)
		if self.weights_dir != '': 
			feature_extractor.load_weights(self.weights_dir)

		if transfer_learning:
			for l in feature_extractor.layers:
				l.trainable = False

		# feature_extractor.summary()
		return feature_extractor

	def normalize(self, reshaped_image):
		def norm(reshaped_image):
			reshaped_image = reshaped_image.astype('float')
			reshaped_image /= 255.
			return reshaped_image
		return norm(reshaped_image)

class YoloMobileNet(BaseNet):
	"""docstring for MobileNet"""
	def __init__(self, net_input_size, anchors, n_class, weights_dir, labels):
		super(YoloMobileNet, self).__init__(net_input_size, anchors, n_class, weights_dir, labels)

	def create_base_network(self, transfer_learning=False):
		model = MobileNet(include_top=False, 
			weights=self.weights_dir, 
			input_shape=(self.net_input_size, self.net_input_size, 3), 
			pooling='avg')
		
		feature_extractor = Model(model.layers[0].input, model.layers[-2].output)

		if transfer_learning:
			for l in feature_extractor.layers:
				l.trainable = False

		# feature_extractor.summary()
		return feature_extractor

	def normalize(self, reshaped_image):
		def norm(reshaped_image):
			reshaped_image = reshaped_image.astype('float')
			reshaped_image /= 255.
			return reshaped_image
		return norm(reshaped_image)

class YoloResNet50(BaseNet):
	"""docstring for ResNet50"""
	def __init__(self, net_input_size, anchors, n_class, weights_dir, labels):
		super(YoloResNet50, self).__init__(net_input_size, anchors, n_class, weights_dir, labels)

	def create_base_network(self, transfer_learning=False):
		model = ResNet50(include_top=False, 
			weights=self.weights_dir, 
			input_shape=(self.net_input_size, self.net_input_size, 3), 
			pooling='avg')
		
		feature_extractor = Model(model.layers[0].input, model.layers[-2].output)

		if transfer_learning:
			for l in feature_extractor.layers:
				l.trainable = False

		# feature_extractor.summary()
		return feature_extractor

	def normalize(self, reshaped_image):
		def norm(reshaped_image):
			reshaped_image = reshaped_image.astype('float')
			reshaped_image /= 255.
			return reshaped_image
		return norm(reshaped_image)

class YoloResNet18(BaseNet):
	"""docstring for ResNet50"""
	def __init__(self, net_input_size, anchors, n_class, weights_dir, labels):
		super(YoloResNet18, self).__init__(net_input_size, anchors, n_class, weights_dir, labels)

	def create_base_network(self, transfer_learning=False):
		ResNet18, _ = Classifiers.get('resnet18')
		model = ResNet18(input_shape=(self.net_input_size, self.net_input_size, 3), weights=self.weights_dir)
		feature_extractor = Model(model.layers[0].input, model.layers[-4].output)

		if transfer_learning:
			for l in feature_extractor.layers:
				l.trainable = False

		# feature_extractor.summary()
		return feature_extractor

	def normalize(self, reshaped_image):
		def norm(reshaped_image):
			reshaped_image = reshaped_image.astype('float')
			reshaped_image /= 255.
			return reshaped_image
		return norm(reshaped_image)