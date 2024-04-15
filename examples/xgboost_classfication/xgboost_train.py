# Importing dataset from sklearn
from sklearn import datasets, metrics

iris = datasets.load_iris()  # dataset loading
X = iris.data  # Features stored in X
y = iris.target  # Class variable

# Splitting dataset into Training (80%) and testing data (20%) using train_test_split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create an XGB classifier and instance of the same
from xgboost import XGBClassifier

clf = XGBClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
# classification accuracy
from sklearn import metrics

print(f"Model accuracy is {metrics.accuracy_score(y_test, y_pred)}")
saved_model = "iris_model.json"
print(f"Saving trained model to {saved_model}")
clf.save_model(saved_model)
