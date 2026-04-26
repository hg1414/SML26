from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

X, y = make_classification(
    n_samples=2000, n_features=10, n_informative=5,
    n_clusters_per_class=3,
    weights=[0.95, 0.05], flip_y=0.02, random_state=42
)

#Splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

#Standard model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
acc = accuracy_score(y_test, y_pred)
print(acc)

#improved model
print("with improve")
model_balanced = LogisticRegression(class_weight='balanced')
model_balanced.fit(X_train, y_train)
y_pred_balanced = model_balanced.predict(X_test)
print(confusion_matrix(y_test, y_pred_balanced))
print(accuracy_score(y_test, y_pred_balanced))

