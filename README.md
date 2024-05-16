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

23:16

</details>