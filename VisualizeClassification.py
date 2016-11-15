import numpy as np
import matplotlib.pyplot as plt


def generate_data_gaussians(centers=((-3, -3), (2, 2)), nr_of_points=100, ratio=0.5, std=1):
  nr_positives = int(nr_of_points*ratio)
  nr_negatives = int(nr_of_points*(1 - ratio))
  gp = np.random.normal(centers[0], scale=std, size=(nr_positives, 2))
  gn = np.random.normal(centers[1], scale=std, size=(nr_positives, 2))
  return np.concatenate([gp, gn], 0), \
         np.concatenate([np.ones((nr_positives,)), np.zeros((nr_negatives,))])


def generate_data_circle(radius=10, nr_of_points=100, ratio=0.5, noise=0.05):
  nr_positives = int(nr_of_points*ratio)
  nr_negatives = int(nr_of_points*(1 - ratio))
  rp = np.random.random((nr_positives,))*radius*0.5
  rn = np.random.uniform(radius*0.7, radius, size=(nr_negatives,))
  r = np.concatenate([rp, rn])
  angle = np.random.uniform(0, 2*np.pi, (nr_of_points))
  x = r * np.sin(angle)
  y = r * np.cos(angle)
  return np.stack([x, y], axis=1),\
         np.concatenate([np.ones((nr_positives,)), np.zeros((nr_negatives,))])

def plot_classification(data, labels, clf):
  colors = ['b', 'r', 'g', 'c', 'm', 'y']
  # create a mesh to plot in
  x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
  y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                       np.arange(y_min, y_max, 0.02))
  Z = clf(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
  for i, c in enumerate(np.unique(labels)):
    c_indices = labels == c
    plt.scatter(data[c_indices, 0], data[c_indices, 1], c=colors[i])

data, labels = generate_data_gaussians()# generate_data_circle()
classifier = lambda x: np.zeros(x.shape[0]) #REPLACE WITH REAL CLASSIFIER


#Make a classifier in tensorflow


plot_classification(data, labels, classifier)
plt.show()
raw_input()