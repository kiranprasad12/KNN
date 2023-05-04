import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

handler = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'Class']

dataset = pd.read_csv(handler, names=names)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.562)

classifer = KNeighborsClassifier(n_neighbors=8)
classifer.fit(x_train, y_train)

y_pre = classifer.predict(x_test)


result = confusion_matrix(y_test, y_pre)
print('confusion_matrix:', result)
result1 = classification_report(y_test, y_pre)
print("classificarion_report:", result1)
result2 = accuracy_score(y_test, y_pre)
print("accuracy_score:", result2)
