### Notes

<details>
<summary>
    
### ML short actions:
</summary>

1. Import the Data
2. Clean the Data
3. Split the Data into Training/Test Sets
4. Create a Model
5. Train the Model
6. Make Predictions
7. Evaluate and Improve
</details>

<details>
<summary>
    
### Libraries and Tools:
</summary>

Numpy
Pandas
MatPlotLib
Scikit-Learn

</details>

<details>
<summary>
    
### Used programms:
</summary>

Install Anaconda
Open Anaconda cmd and type `jupyter notebook`.
Go to `http://localhost:8888/tree`.
Create file and now u can run it.

</details>

<details>
<summary>
    
### Importing Data Set:
</summary>

`https://www.kaggle.com`
For first learn I use:
`https://www.kaggle.com/datasets/gregorut/videogamesales`
Download it and get zip file.

`import pandas as pd` import and rename pandas lib to pd
`df = pd.read_csv('vgsales.csv')` load csv file
^ note: `vgsales.csv` file in same path

#### `pandas` usefull methods:

`df.shape` - show (records, columns)

`df.describe()` - show grouped table data

`df.values` - show values in list(array)

#### `jupyter` shortcuts

`d + d` - delete selected row
`a` - add above row
`b` - add behind row
`tab` - show all methods
`shift + tab` - show method signature
`ctrl + /` - comment/uncomment row

</details>

<details>
<summary>
    
### Adding music csv file:
</summary>

1. Import the Data

At first need import pandas `import pandas as pd`
then need set as variable `music_data = pd.read_csv('music.csv')`

2. Clean the Data
   We dont need clean data because it's already clean

After this, we also need to divide our database into two categories
input and output dataset. To implement this we will use the `.drop()` method
This method allows you to remove unnecessary columns. (It does not change the original data but actually creates a new database but without the selected columns)
Therefore, by common convention, such data is designated with a capital letter `X`

Now we must create output dataset and by common convention, such data is desigated
with a lowercase letter `y` `y = music_data['genre']`

The next step is a build model by using ML algorithm. In this time we will use a
simple algorithm calling design tree in library `scikit-learn`

`from sklearn.tree import DecisionTreeClassifier`

after this we set new object to `DecisionTreeClassifier` class and call his
`fit` method. That method take 2 parametrs: input and output dataset

```py
model = DecisionTreeClassifier()
```

```py
model.fit(X, y)
```

to get predictions using `DecisionTreeClassifier` we need call `predict`
method from our model.

```py
predictions = model.predict([ [21, 1], [22, 0] ])
```

but that one is a old version, here are new version

```py
# Данные для предсказания, оформленные как DataFrame
prediction_data = pd.DataFrame({
    'age': [21, 22],
    'gender': [1, 0]
})

# Предсказание
predictions = model.predict(prediction_data)
print(predictions)
```

now we need calculate our model Accuracy

```py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(X, y)

predictions = model.predict([ [21, 1], [22, 0] ])
predictions
```

Calculating Accuracy

```py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split  # easy split our dataset to 2 sets (training and setting)

from sklearn.metrics import accuracy_score  # class to detect our accuracy score

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # get 3 arguments: input and output dataset and  set percent to testing (0.2 == 20%)
# This method return a tuple fitst two is an input sets for training and second two is an output sets for training

model = DecisionTreeClassifier()
# Now to start training we want send out training dataset
model.fit(X_train, y_train)  # model.fit(X, y)
# also we past here X_test (), this dataset contains the input values for testing
predictions = model.predict(X_test)  # actually values
# to calculate our accuracy we just need to compare with our actual y_test values
score = accuracy_score(y_test, predictions)  # its contained accepted values and actually values

score
```

To test our Accuracy score we can press ctrl + Enter and we rerun current block multiple times. This class always get randomly values from our database.

And if we set our testing size to 0.8 (80%) its means that we use 20% our dataset to training and 80% to testing

</details>

<details>
<summary>
    
### Persisting Models (Сохраняющиеся модели):
</summary>

Чтобы не создавать каждый раз нашу модель для каждого нового пользователя нам необходимо где то сохранить уже созданные модели.

```py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # используется для выполнения задач классификации

import joblib  # импортируем joblib обект. Этот обект тиеет методы для сохранения наших моделей

# Теперь чтобы каждый раз не пересобрать нашу модель прокоментируем наш код
# music_data = pd.read_csv('music.csv')
# X = music_data.drop(columns=['genre'])
# y = music_data['genre']

# model = DecisionTreeClassifier()  #
# model.fit(X, y)  # обучаем модель

# После обучения вызываем и передаем два аргументы
# joblib.dump(model, 'music-recommender.joblib')  # получает модель и название файла где хранит

# predictions = model.predict([21, 1])  # Временно закоментирую строку прогнозов

# Для загрузки сохраненной дамп файла
model = joblib.load('music-recommender.joblib')
predictions = model.predict([[21, 1]])
predictions
```

</details>

<details>
<summary>
    
### Visualizing Decision Trees (Визуализация деревьев решений):
</summary>

```py
# Упрощаем код для визуализации
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# обект tree имеет метод для вывода в графическом формате
from sklearn import tree

# Импортируем набор данных
music_data = pd.read_csv('music.csv')
# Создаем наборы входных и выходных данных (imput and outout datasets)
X = music_data.drop(columns=['genre'])
y = music_data['genre']

# Создаем модель
model = DecisionTreeClassifier()
# Обучаем
model.fit(X, y)

# После обучения модели вызываем метод для создания дот файла
tree.export_graphviz(model, out_file='music-recommender.dot',
                            feature_names=['age', 'gender'],
                            class_names=sorted(y.unique()),
                            label = 'all',
                            rounded=True,
                            filled=True)
```

Для визуализации .dot формата в VScode надо установить Graphviz (dot)
filled=True - красит наши блоки в разные цвета
rounded=True - округляет угол квадратов
label = 'all' - каждая секция будет иметь текстовое описание
class_names=sorted(y.unique()) - отображает классы используя уникальные жанры
feature_names=['age', 'gender'], - Устанавливаем по каким критериям происходит сравнение правила

</details>
