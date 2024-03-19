from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Dictionary of models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=1),
    "AdaBoost": AdaBoostClassifier(n_estimators=100),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100),
    "Bagging Classifier": BaggingClassifier(n_estimators=100),
    "SVC": SVC(probability=True),
    "Linear SVC": LinearSVC(max_iter=10000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
    "Ridge Classifier": RidgeClassifier(),
    "SGD Classifier": SGDClassifier(max_iter=1000, tol=1e-3),
    "MLP Classifier": MLPClassifier(max_iter=1000),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0)  # verbose=0 to silence CatBoost output
}

# Evaluate each model
for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"{name} accuracy: {np.mean(cv_scores):.4f}")

# Choose the best model based on cross-validation accuracy
best_model_name, best_model = max(models.items(), key=lambda x: np.mean(cross_val_score(x[1], X, y, cv=5, scoring='accuracy')))
best_model_accuracy = np.mean(cross_val_score(best_model, X, y, cv=5, scoring='accuracy'))
print(f"\nBest model: {best_model_name} with accuracy: {best_model_accuracy:.4f}")

# Fit the best model on the full training data and prepare for submission
best_model.fit(X, y)
X_test = pd.get_dummies(test_data[features])
predictions = best_model.predict(X_test)

