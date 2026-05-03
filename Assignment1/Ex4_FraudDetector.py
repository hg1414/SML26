from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

X, y = make_classification(
    n_samples=2000, n_features=10, n_informative=5,
    n_clusters_per_class=3,
    weights=[0.95, 0.05], flip_y=0.02, random_state=42
)

#splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

#standard model
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

#scores/probs fraud
y_score = model.predict_proba(X_test)[:, 1]
y_score_balanced = model_balanced.predict_proba(X_test)[:, 1]

#ROC-curve
fpr, tpr, thresholds = roc_curve(y_test, y_score)
fpr_balanced, tpr_balanced, thresholds_balanced = roc_curve(y_test, y_score_balanced)

#AUC
auc = roc_auc_score(y_test, y_score)
auc_balanced = roc_auc_score(y_test, y_score_balanced)

print("AUC standard model:", auc)
print("AUC balanced model:", auc_balanced)

#plot
plt.figure()
plt.plot(fpr, tpr, label=f"Standard model, AUC = {auc:.3f}")
plt.plot(fpr_balanced, tpr_balanced, label=f"Balanced model, AUC = {auc_balanced:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")

plt.xlabel("FP Rate")
plt.ylabel("TP Rate / Recall")
plt.title("ROC Curve")
plt.legend()
plt.show()

from sklearn.metrics import precision_recall_curve, average_precision_score

# Precision-Recall für Standardmodell
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_score)

# Precision-Recall für Balanced Modell
precision_bal, recall_bal, thresholds_pr_bal = precision_recall_curve(y_test, y_score_balanced)

# Average Precision (ähnlich wie AUC für PR)
ap = average_precision_score(y_test, y_score)
ap_bal = average_precision_score(y_test, y_score_balanced)

print("Average Precision standard:", ap)
print("Average Precision balanced:", ap_bal)

# Plot
plt.figure()
plt.plot(recall, precision, label=f"Standard model, AP = {ap:.3f}")
plt.plot(recall_bal, precision_bal, label=f"Balanced model, AP = {ap_bal:.3f}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig("plot2.png")
plt.show()
