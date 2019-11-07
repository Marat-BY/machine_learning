# Иллюстрация метода KNN (K Nearest Neighbours)
# Работа разделена на две части
# 1. Формирование данных при помощи генератора случайных чисел
# 2. Построение модели KNN на сформированных данных

# Формирование данных для модели
import random
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap


def randomClusterGenerator(xPoint, yPoint, pointsInCluster, sigma, clusterNumber):
	"""
		Функция randomClusterGenerator генерирует набор случайных, нормально
		распределенных точек (pointsInCluster) вокруг центра точки Х и У со 
		стандартным отклонением sigma. Если необходимо, пользователь указывает
		номер кластера clusterNumber.

		Пример использования данной функции:

		randomClusterGenerator(20, 25, 5, 1, 1)

		output: 

		[
		  (20.1385332, 21.123955),
		  (22.2458811, 22.588934),
		  (24.1098219, 23.218577),
		  (22.3958811, 21.295865),
		  (21.3120991, 21.235996)
		]
	"""
	clusterData = []
	for point in range(pointsInCluster):
		clusterData.append((random.gauss(xPoint, sigma), random.gauss(yPoint, sigma),clusterNumber))

	return clusterData


def generateNCluster(clusterNumber, minCoordinate, maxLength, pointsInCluster, sigma):
	"""
	  Генерация N-количества кластеров, количество указывается в параметре
	  clusterNumber, в пределах координатной плоскости (х, у), начиная от 
	  параметра minCoordinate до minCoordinate + maxLength.
	"""

	clusterData = []
	for cluster in range(clusterNumber):
		clusterData.append(randomClusterGenerator (minCoordinate + maxLength + random.random(),
												   minCoordinate + maxLength + random.random(),
												   pointsInCluster,
												   sigma,
												   cluster))
	return clusterData

def drawClusters(clusterData):
	"""
		Набор кластеров по данным из clusterData
	"""
	for cluster in clusterData:
		xData.append(point[0])
		yData.append(point[1])
		colors.append(point[2])

	plt.scatter(xData, yData,  label = colors[0])
	plt.legend(loc = 'upper right')
	plt.show();

# Parameters
clusterNumber = 3
minCoordinate = 0
maxLength = 100
pointsInCluster = 15
sigma = 5

data = generateNCluster(clusterNumber, minCoordinate, maxLength, pointsInCluster, sigma)
drawClusters(data)

# Построение модели KNN
# # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean', p = 2)

X = []
Y = []

for cluster in data:
	for point in cluster:
		X.append([point[0], point[1]])
		Y.append(point[2])

model.fit(X, Y)

def KNeighbors(clusterData, model):
	"""
		Визуализация данных классификации модели KNN
	"""
	step = 1
	XX, yy = np.meshgrid(np.arange(minCoordinate, minCoordinate + maxLength, step),
						 np.arange(minCoordinate, minCoordinate + maxLength, step))

	Z = model.predict(np.c_[XX.ravel(), yy.ravel()])
	Z = Z.reshape(XX.shape)

	cmpa_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
	plt.pcolormesh(XX, yy, Z, cmap = cmpa_light)

	for cluster in clusterData:
		xData = []
		yData = []
		colors = []

		for point in cluster:
			xData.append(point[0])
			yData.append(point[1])
			colors.append(point[2])

		plt.scatter(xData, yData, label = colors[0])
	plt.legend(loc = 'upper right')
	plt.show();

data = generateNCluster(clusterNumber, minCoordinate, maxLength, pointsInCluster, sigma)
X = []
Y = []

for cluster in data:
	for point in cluster:
		X.append(point[0], point[1])
		Y.append(point[2])

model.fit(X, Y)

KNeighbors(data, model)