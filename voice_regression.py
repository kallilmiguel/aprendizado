# Importing the libraries
import numpy as np
import pandas as pd

# Lista de Parametros
RATIO_TESTE_TREINO = 1000/3168
rs = np.random.randint(0,100)


#Importando a base de dados
dataset = pd.read_csv('voice.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 20].values

#  Transformando as labels em numeros
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
onehotencoder = OneHotEncoder(categorical_features = [0])
y = onehotencoder.fit_transform(y).toarray()
y = np.transpose(y)

# Dividindo a base de dados em treinamento e teste
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=RATIO_TESTE_TREINO, stratify=y)

# Metodo de regressao linear simples
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Prevendo os resultados da base de testes
y_pred = regressor.predict(X_test)
y_pred = np.round(y_pred)

#Calcular a precisao dos resultados
from sklearn.metrics import accuracy_score
print('Accuracy simple linear regression: %.2f\n' % accuracy_score(y_test, y_pred))