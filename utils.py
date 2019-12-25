import os
import xml.etree.ElementTree as ET
import numpy as np
import copy
import cv2
from imgaug import augmenters as iaa
from boundbox import BoundBox

def _interval_overlap(interval_a, interval_b):
	x1, x2 = interval_a
	x3, x4 = interval_b

	if x3 < x1:
		if x4 < x1:
			return 0
		else:
			return min(x2,x4) - x1
	else:
		if x2 < x3:
			 return 0
		else:
			return min(x2,x4) - x3          

def _sigmoid(x):
	return 1. / (1. + np.exp(-x))

def _softmax(x, axis=-1, t=-100.):
	x = x - np.max(x)
	
	if np.min(x) < t:
		x = x/np.min(x)*t
		
	e_x = np.exp(x)
	
	return e_x / e_x.sum(axis, keepdims=True)


def bbox_iou(box1, box2):
	intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
	intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
	
	intersect = intersect_w * intersect_h

	w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
	w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
	
	union = w1*h1 + w2*h2 - intersect
	
	return float(intersect) / union

def parse_annotation(ann_dir, img_dir, labels=[]):
	all_imgs = []
	seen_labels = {}
	
	# limit = 16
	
	for ann in sorted(os.listdir(ann_dir)):
		# if len(all_imgs) == limit:
		#     break
		img = {'object':[]}
		
		tree = ET.parse(ann_dir + ann)
    
		for elem in tree.iter():
			if 'filename' in elem.tag:
				img['filename'] = img_dir + elem.text
			if 'width' in elem.tag:
				img['width'] = int(elem.text)
			if 'height' in elem.tag:
				img['height'] = int(elem.text)
			if 'object' in elem.tag or 'part' in elem.tag:
				obj = {}
				
				for attr in list(elem):
					if 'name' in attr.tag:
						obj['name'] = attr.text

						if obj['name'] in seen_labels:
							seen_labels[obj['name']] += 1
						else:
							seen_labels[obj['name']] = 1
						
						if len(labels) > 0 and obj['name'] not in labels:
							break
						else:
							img['object'] += [obj]
							
					if 'bndbox' in attr.tag:
						for dim in list(attr):
							if 'xmin' in dim.tag:
								obj['xmin'] = int(round(float(dim.text)))
							if 'ymin' in dim.tag:
								obj['ymin'] = int(round(float(dim.text)))
							if 'xmax' in dim.tag:
								obj['xmax'] = int(round(float(dim.text)))
							if 'ymax' in dim.tag:
								obj['ymax'] = int(round(float(dim.text)))

		if len(img['object']) > 0:
			all_imgs += [img]
						
	return all_imgs, seen_labels


def generate_Xy(imgs, labels, anchors, n_grid, net_input_size, n_class, normalize, aug=True):
	
	# input images
	X_train = np.zeros((len(imgs), net_input_size, net_input_size, 3))
	# desired network output
	y_train = np.zeros((len(imgs), n_grid, n_grid, len(anchors)//2, 4 + 1 + n_class))
	
	
	anchor_boxes = [BoundBox(0, 0, anchors[2*i], anchors[2*i+1]) for i in range(int(len(anchors)//2))]
	
	n = 0
	
	for img in imgs:
		
		reshaped_image, all_objects = aug_image(img, net_input_size, aug=aug)
		
		# image_name = img['filename']
		# if '.jpg' not in image_name and '.png' not in image_name:
		# 	image_name += '.jpg'

		# image = cv2.imread(image_name)
		# reshaped_image = cv2.resize(image, (net_input_size, net_input_size))
		# reshaped_image = reshaped_image[:,:,::-1]
		
		# all_objects = img['object']

		reshaped_image = normalize(reshaped_image)
		
		X_train[n] = reshaped_image
		
		for obj in all_objects:
			if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in labels:
				
				# unit: grid cell
				center_x = .5*(obj['xmin'] + obj['xmax'])
				center_x = center_x / (float(net_input_size) / n_grid)
				
				center_y = .5*(obj['ymin'] + obj['ymax'])
				center_y = center_y / (float(net_input_size) / n_grid)

				# grid_x: row, grid_y: column
				grid_x = int(np.floor(center_x))
				grid_y = int(np.floor(center_y))

				if grid_x < n_grid and grid_y < n_grid:
					obj_indx  = labels.index(obj['name'])
					
					# unit: grid cell
					center_w = (obj['xmax'] - obj['xmin']) / (float(net_input_size) / n_grid) 
					center_h = (obj['ymax'] - obj['ymin']) / (float(net_input_size) / n_grid) 

					box = [center_x, center_y, center_w, center_h]
					
					# find the anchor that best predicts this box
					best_anchor = -1
					max_iou     = -1

					shifted_box = BoundBox(0, 
										   0,
										   center_w,                                                
										   center_h)

					for i in range(len(anchor_boxes)):
						anchor = anchor_boxes[i]
						iou    = bbox_iou(shifted_box, anchor)

						if max_iou < iou:
							best_anchor = i
							max_iou     = iou

					# assign ground truth x, y, w, h, confidence and class probs to y_batch
					y_train[n, grid_y, grid_x, best_anchor, 0:4] = box
					y_train[n, grid_y, grid_x, best_anchor, 4  ] = 1.
					y_train[n, grid_y, grid_x, best_anchor, 5 + obj_indx] = 1
					
		n += 1

	return X_train, y_train


sometimes = lambda aug: iaa.Sometimes(0.5, aug)

aug_pipe = iaa.Sequential(
			[
				# apply the following augmenters to most images
				#iaa.Fliplr(0.5), # horizontally flip 50% of all images
				#iaa.Flipud(0.2), # vertically flip 20% of all images
				#sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
				sometimes(iaa.Affine(
					#scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
					#translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
					#rotate=(-5, 5), # rotate by -45 to +45 degrees
					#shear=(-5, 5), # shear by -16 to +16 degrees
					#order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
					#cval=(0, 255), # if mode is constant, use a cval between 0 and 255
					#mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
				)),
				# execute 0 to 5 of the following (less important) augmenters per image
				# don't execute all of them, as that would often be way too strong
				iaa.SomeOf((0, 5),
					[
						#sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
						iaa.OneOf([
							iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
							iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
							iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
						]),
						iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
						#iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
						# search either for all edges or for directed edges
						#sometimes(iaa.OneOf([
						#    iaa.EdgeDetect(alpha=(0, 0.7)),
						#    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
						#])),
						iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
						iaa.OneOf([
							iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
							#iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
						]),
						#iaa.Invert(0.05, per_channel=True), # invert color channels
						iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
						iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
						iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
						#iaa.Grayscale(alpha=(0.0, 1.0)),
						#sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
						#sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
					],
					random_order=True
				)
			],
			random_order=True
		)

def aug_image(img, net_input_size, aug):
	image_name = img['filename']

	image_name = img['filename']
	if '.jpg' not in image_name and '.png' not in image_name:
		image_name += '.jpg'

	image = cv2.imread(image_name)

	if image is None: print('Cannot find ', image_name)

	h, w, c = image.shape
	all_objs = copy.deepcopy(img['object'])

	if aug:
		### scale the image
		scale = np.random.uniform() / 10. + 1.
		image = cv2.resize(image, (0,0), fx = scale, fy = scale)

		### translate the image
		max_offx = (scale-1.) * w
		max_offy = (scale-1.) * h
		offx = int(np.random.uniform() * max_offx)
		offy = int(np.random.uniform() * max_offy)
		
		image = image[offy : (offy + h), offx : (offx + w)]

		### flip the image
		flip = np.random.binomial(1, .5)
		if flip > 0.5: image = cv2.flip(image, 1)
			
		image = aug_pipe.augment_image(image)            
		
	# resize the image to standard size
	image = cv2.resize(image, (net_input_size, net_input_size))
	image = image[:,:,::-1]

	# fix object's position and size
	for obj in all_objs:
		for attr in ['xmin', 'xmax']:
			if aug: obj[attr] = int(obj[attr] * scale - offx)
				
			obj[attr] = int(obj[attr] * float(net_input_size) / w)
			obj[attr] = max(min(obj[attr], net_input_size), 0)
			
		for attr in ['ymin', 'ymax']:
			if aug: obj[attr] = int(obj[attr] * scale - offy)
				
			obj[attr] = int(obj[attr] * float(net_input_size) / h)
			obj[attr] = max(min(obj[attr], net_input_size), 0)

		if aug and flip > 0.5:
			xmin = obj['xmin']
			obj['xmin'] = net_input_size - obj['xmax']
			obj['xmax'] = net_input_size - xmin
			
	return image, all_objs


def decode_netout(netout, anchors, nb_class, obj_threshold=0.1, nms_threshold=0.1):
	grid_h, grid_w, nb_box = netout.shape[:3]

	boxes = []
	
	# decode the output by the network
	netout[..., 4]  = _sigmoid(netout[..., 4])
	netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
	netout[..., 5:] *= netout[..., 5:] > obj_threshold
	
	for row in range(grid_h):
		for col in range(grid_w):
			for b in range(nb_box):
				# from 4th element onwards are confidence and class classes
				classes = netout[row,col,b,5:]
				
				if np.sum(classes) > 0:
					# first 4 elements are x, y, w, and h
					x, y, w, h = netout[row,col,b,:4]

					x = (col + _sigmoid(x)) / grid_w # center position, unit: image width
					y = (row + _sigmoid(y)) / grid_h # center position, unit: image height
					w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
					h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
					confidence = netout[row,col,b,4]
					
					box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, confidence, classes)
					
					boxes.append(box)

	# suppress non-maximal boxes
	for c in range(nb_class):
		sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

		for i in range(len(sorted_indices)):
			index_i = sorted_indices[i]
			
			if boxes[index_i].classes[c] == 0: 
				continue
			else:
				for j in range(i+1, len(sorted_indices)):
					index_j = sorted_indices[j]
					
					if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
						boxes[index_j].classes[c] = 0
						
	# remove the boxes which are less likely than a obj_threshold
	boxes = [box for box in boxes if box.get_score() > obj_threshold]
	
	return boxes


def load_annotation(imgs, i, labels):
		annots = []

		for obj in imgs[i]['object']:
			annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], labels.index(obj['name'])]
			annots += [annot]

		if len(annots) == 0: annots = [[]]

		return np.array(annots)

def compute_overlap(a, b):
	"""
	Code originally from https://github.com/rbgirshick/py-faster-rcnn.
	Parameters
	----------
	a: (N, 4) ndarray of float
	b: (K, 4) ndarray of float
	Returns
	-------
	overlaps: (N, K) ndarray of overlap between boxes and query_boxes
	"""
	area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

	iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
	ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

	iw = np.maximum(iw, 0)
	ih = np.maximum(ih, 0)

	ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

	ua = np.maximum(ua, np.finfo(float).eps)

	intersection = iw * ih

	return intersection / ua  
	
def compute_ap(recall, precision):
	""" Compute the average precision, given the recall and precision curves.
	Code originally from https://github.com/rbgirshick/py-faster-rcnn.

	# Arguments
		recall:    The recall curve (list).
		precision: The precision curve (list).
	# Returns
		The average precision as computed in py-faster-rcnn.
	"""
	# correct AP calculation
	# first append sentinel values at the end
	mrec = np.concatenate(([0.], recall, [1.]))
	mpre = np.concatenate(([0.], precision, [0.]))

	# compute the precision envelope
	for i in range(mpre.size - 1, 0, -1):
		mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

	# to calculate area under PR curve, look for points
	# where X axis (recall) changes value
	i = np.where(mrec[1:] != mrec[:-1])[0]

	# and sum (\Delta recall) * prec
	ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
	return ap      