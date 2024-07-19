# Running_Multiple_Regressor_At_Once
# The library you're referring to is likely **TPOT** (Tree-based Pipeline Optimization Tool). TPOT is a Python library that uses genetic programming to optimize machine learning pipelines. It automates the process of selecting the best model and its hyperparameters by evaluating multiple models and their combinations.

Here's a basic example of how to use TPOT:

```python
from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Load data
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=42)

# Create and train TPOT regressor
tpot = TPOTRegressor(verbosity=2, generations=5, population_size=50)
tpot.fit(X_train, y_train)

# Evaluate the best model
print(tpot.score(X_test, y_test))

# Export the best model
tpot.export('best_model.py')
```

In this example, TPOT will automatically test and optimize various regression models and their hyperparameters on the Boston housing dataset, and it will export the best model pipeline found.
Another library that allows running multiple models for comparison is **H2O.ai**. H2O provides a tool called AutoML that automates the process of training and tuning a large selection of models within a specified time limit.

Here’s a basic example of using H2O’s AutoML for regression:

```python
import h2o
from h2o.automl import H2OAutoML

# Initialize H2O cluster
h2o.init()

# Load data
data = h2o.import_file("path/to/your/data.csv")

# Define predictors and response
x = data.columns[:-1]  # All columns except the last one
y = data.columns[-1]   # The last column

# Split data into train and test
train, test = data.split_frame(ratios=[.8], seed=1234)

# Run AutoML
aml = H2OAutoML(max_runtime_secs=3600, seed=1)
aml.train(x=x, y=y, training_frame=train)

# View the AutoML Leaderboard
lb = aml.leaderboard
print(lb)

# Get the best model
best_model = aml.leader
performance = best_model.model_performance(test)
print(performance)
```

Both TPOT and H2O's AutoML can significantly speed up the process of finding the best model for your regression tasks.
