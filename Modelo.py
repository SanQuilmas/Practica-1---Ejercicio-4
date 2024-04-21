import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

flag = True

num_epochs = 1000

model = nn.Sequential(
	nn.Linear(4, 120),
	nn.Tanh(),
	nn.Linear(120, 48),
	nn.Tanh(),
	nn.Linear(48, 12),
	nn.Tanh(),
	nn.Linear(12, 3),
	nn.Sigmoid()
)

loss_fn = nn.BCELoss()

loss_values = []

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Cargando archivo de entrenamiento...")
df = pd.read_csv('irisbin.csv', header=None)

#-----------------------------------------------------------------------------------------------------------------------

X = df.iloc[0:100, [0, 1, 2, 3]].values
y = df.iloc[0:100, [4, 5, 6]].values
y = np.where(y == -1, 0, 1)
print("K 1/3")
print("Generando Tensores...")

X_tensor = torch.tensor(X, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.float)

#print("Revisando si existe modelo previo...")

my_file = Path("model.pickle")
if not my_file.is_file():
	#print("Modelo no encontrado...")
	print("Entrenando Modelo con Archivo de Entrenamiento...")
	for epoch in range(num_epochs):
		y_pred = model(X_tensor)
		loss = loss_fn(y_pred, y_tensor)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		#if (epoch + 1) % 100 == 0:
			#print(f'Epoch [{epoch+1}/{num_epochs}], Perdida: {loss.item():.4f}')
	#torch.save(model, "model.pickle")
	flag = False
else:
	print("Modelo encontrado...")
	model = torch.load("model.pickle")

print("Cargando archivo de Prueba...")
print("Probando Modelo con Archivo de Prueba...")

X = df.iloc[100:150, [0, 1, 2, 3]].values
y = df.iloc[100:150, [4, 5, 6]].values
y = np.where(y == -1, 0, 1)

X_tensor = torch.tensor(X, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.float)

y_pred = model(X_tensor)
loss = loss_fn(y_pred, y_tensor)
y_pred = np.where(y_pred < 0.5, 0, 1)
print(f'Perdida: {loss.item():.4f}')
loss_values.append(loss.item())

#-----------------------------------------------------------------------------------------------------------------------

X = df.iloc[50:150, [0, 1, 2, 3]].values
y = df.iloc[50:150, [4, 5, 6]].values
y = np.where(y == -1, 0, 1)
print("K 2/3")
print("Generando Tensores...")

X_tensor = torch.tensor(X, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.float)

#print("Revisando si existe modelo previo...")

my_file = Path("model.pickle")
if not my_file.is_file():
	#print("Modelo no encontrado...")
	print("Entrenando Modelo con Archivo de Entrenamiento...")
	for epoch in range(num_epochs):
		y_pred = model(X_tensor)
		loss = loss_fn(y_pred, y_tensor)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		#if (epoch + 1) % 100 == 0:
			#print(f'Epoch [{epoch+1}/{num_epochs}], Perdida: {loss.item():.4f}')
	#torch.save(model, "model.pickle")
	flag = False
else:
	print("Modelo encontrado...")
	model = torch.load("model.pickle")

print("Cargando archivo de Prueba...")
print("Probando Modelo con Archivo de Prueba...")

X = df.iloc[0:50, [0, 1, 2, 3]].values
y = df.iloc[0:50, [4, 5, 6]].values
y = np.where(y == -1, 0, 1)

X_tensor = torch.tensor(X, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.float)

y_pred = model(X_tensor)
loss = loss_fn(y_pred, y_tensor)
y_pred = np.where(y_pred < 0.5, 0, 1)
print(f'Perdida: {loss.item():.4f}')
loss_values.append(loss.item())

#-----------------------------------------------------------------------------------------------------------------------

X = df.iloc[0:50 + 100:150, [0, 1, 2, 3]].values
y = df.iloc[0:50 + 100:150, [4, 5, 6]].values
y = np.where(y == -1, 0, 1)
print("K 3/3")
print("Generando Tensores...")

X_tensor = torch.tensor(X, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.float)

#print("Revisando si existe modelo previo...")

my_file = Path("model.pickle")
if not my_file.is_file():
	#print("Modelo no encontrado...")
	print("Entrenando Modelo con Archivo de Entrenamiento...")
	for epoch in range(num_epochs):
		y_pred = model(X_tensor)
		loss = loss_fn(y_pred, y_tensor)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		#if (epoch + 1) % 100 == 0:
			#print(f'Epoch [{epoch+1}/{num_epochs}], Perdida: {loss.item():.4f}')
	#torch.save(model, "model.pickle")
	flag = False
else:
	print("Modelo encontrado...")
	model = torch.load("model.pickle")

print("Cargando archivo de Prueba...")
print("Probando Modelo con Archivo de Prueba...")

X = df.iloc[50:100, [0, 1, 2, 3]].values
y = df.iloc[50:100, [4, 5, 6]].values
y = np.where(y == -1, 0, 1)

X_tensor = torch.tensor(X, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.float)

y_pred = model(X_tensor)
loss = loss_fn(y_pred, y_tensor)
y_pred = np.where(y_pred < 0.5, 0, 1)
print(f'Perdida: {loss.item():.4f}')
loss_values.append(loss.item())

#-----------------------------------------------------------------------------------------------------------------------

avg = 0
for i in loss_values:
	avg = avg + i
avg = avg/len(loss_values)

print('Error leave-k-out: {0:.3f}.'.format(avg))

#-----------------------------------------------------------------------------------------------------------------------
train_start_1 = 0
train_end_1 = 150
train_start_2 = 0
test = -1

loss_values = []

print("...")
print("...")
print("...")

for i in range(150):
	X = df.iloc[train_start_1:train_end_1 + train_start_2, [0, 1, 2, 3]].values
	y = df.iloc[train_start_1:train_end_1 + train_start_2, [4, 5, 6]].values
	y = np.where(y == -1, 0, 1)
	print(f'Leave-one-out {i}/150')
	print("Generando Tensores...")

	X_tensor = torch.tensor(X, dtype=torch.float)
	y_tensor = torch.tensor(y, dtype=torch.float)

	#print("Revisando si existe modelo previo...")

	my_file = Path("model.pickle")
	if not my_file.is_file():
		#print("Modelo no encontrado...")
		print("Entrenando Modelo con Archivo de Entrenamiento...")
		for epoch in range(num_epochs):
			y_pred = model(X_tensor)
			loss = loss_fn(y_pred, y_tensor)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			#if (epoch + 1) % 100 == 0:
				#print(f'Epoch [{epoch+1}/{num_epochs}], Perdida: {loss.item():.4f}')
		#torch.save(model, "model.pickle")
		flag = False
	else:
		print("Modelo encontrado...")
		model = torch.load("model.pickle")

	print("Cargando archivo de Prueba...")
	print("Probando Modelo con Archivo de Prueba...")

	X = df.iloc[test, [0, 1, 2, 3]].values
	y = df.iloc[test, [4, 5, 6]].values
	y = np.where(y == -1, 0, 1)

	X_tensor = torch.tensor(X, dtype=torch.float)
	y_tensor = torch.tensor(y, dtype=torch.float)

	y_pred = model(X_tensor)
	loss = loss_fn(y_pred, y_tensor)
	y_pred = np.where(y_pred < 0.5, 0, 1)
	print(f'Perdida: {loss.item():.10f}')
	loss_values.append(loss.item())

	train_end_1 = train_end_1 - 1
	train_start_2 = train_start_2 - 1
	test = test - 1

#-----------------------------------------------------------------------------------------------------------------------

avg = 0
for i in loss_values:
	avg = avg + i
avg = avg/len(loss_values)

print('Error leave-one-out: {0:.3f}.'.format(avg))

#-----------------------------------------------------------------------------------------------------------------------

if flag:

	X = df.iloc[0:150, [0, 1, 2, 3]].values
	y = df.iloc[0:150, [4, 5, 6]].values
	y = np.where(y == -1, 0, 1)

	X_tensor = torch.tensor(X, dtype=torch.float)
	y_tensor = torch.tensor(y, dtype=torch.float)

	y_pred = model(X_tensor)
	loss = loss_fn(y_pred, y_tensor)
	y_pred = np.where(y_pred < 0.5, 0, 1)

	print("Graficando resultados...")
	color = []

	#for i in y_tensor.tolist():
	for i in y_pred.tolist():
		if i == [0, 0, 1]:
			color.append("#0000FF")
		elif i == [0, 1, 0]:
			color.append("#FF0000")
		elif i == [1, 0, 0]:
			color.append("#00FF00")
	
	x_values = X[:, 2]
	y_values = X[:, 1]
	z_values = X[:, 0]

	plt.scatter(x_values, y_values, z_values, c=color)
	plt.xlabel("x")
	plt.ylabel("y")
	plt.legend()
	plt.show()
else:
	print("Correr el programa de nuevo para ver grafica de resultados...")