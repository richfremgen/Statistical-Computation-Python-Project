import pickle

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import auc, roc_curve, RocCurveDisplay
from main import make_angle, make_distance

dt_shots = pd.read_pickle("data/processed/dt_shots.pkl")

dt_shots = dt_shots[[
    'start_x', 'start_y', 'label_Left', 'label_Right', 'label_counter_attack',
    'label_head/body', 'label_interception', 'angle', 'distance', 'label_Goal'
]].dropna()

dt_shots = dt_shots[dt_shots['start_x'] > 70*105.0/100.0]

X_train, X_test, y_train, y_test = train_test_split(
    dt_shots.drop('label_Goal', axis=1), dt_shots.label_Goal, test_size=0.2,
    random_state=42,
    stratify=dt_shots.label_Goal
)

# Initial cross validation
# param_grid = {
#     'n_estimators': [100, 200, 500, 1000],
#     'min_samples_split': [10, 20, 50],
#     'max_features': [2, 3, 5]
# }
#
# random_forest = RandomForestClassifier(
#     criterion='entropy',
#     n_jobs=2,
#     random_state=42
# )
#
# grid_cv = GridSearchCV(
#     estimator=random_forest,
#     param_grid=param_grid,
#     scoring='roc_auc_ovr',
#     n_jobs=2,
#     cv=StratifiedKFold(n_splits=5),
#     verbose=4
# )
#
# cv_fit = grid_cv.fit(X_train, y_train)

# cv_fit.best_estimator_

# This cross validation is based on the insight provided by the previous one
# param_grid = {
#     'min_samples_split': [50, 100, 200],
#     'max_features': [5, 7, 9]
# }
#
# random_forest = RandomForestClassifier(
#     criterion='entropy',
#     n_estimators=1000,
#     n_jobs=2,
#     random_state=42
# )
#
# grid_cv = GridSearchCV(
#     estimator=random_forest,
#     param_grid=param_grid,
#     scoring='roc_auc_ovr',
#     n_jobs=2,
#     cv=StratifiedKFold(n_splits=5),
#     verbose=4
# )
#
# cv_fit = grid_cv.fit(X_train, y_train)
#
# cv_fit.best_estimator_

# param_grid = {
#     'min_samples_split': [500, 1000],
# }
#
# random_forest = RandomForestClassifier(
#     criterion='entropy',
#     n_estimators=1000,
#     max_features=9,
#     n_jobs=2,
#     random_state=42
# )
#
# grid_cv = GridSearchCV(
#     estimator=random_forest,
#     param_grid=param_grid,
#     scoring='roc_auc_ovr',
#     n_jobs=2,
#     cv=StratifiedKFold(n_splits=5),
#     verbose=4
# )
#
# cv_fit = grid_cv.fit(X_train, y_train)
#
# cv_fit.best_estimator_
# min_samples_split=500

final_rf = RandomForestClassifier(
    criterion='entropy',
    n_estimators=1000,
    max_features=9,
    n_jobs=2,
    min_samples_split=500,
    random_state=42,
    verbose=4
)

final_rf_fit = final_rf.fit(X_train, y_train)

# Save random forest model
# with open('models/ran_forest.pkl', 'wb') as f:
#    pickle.dump(final_rf_fit, f)

probs = final_rf_fit.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, probs[:, 1])
roc_auc = auc(fpr, tpr)
disp = RocCurveDisplay(
    fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Random Forest'
).plot()
plt.title("ROC AUC")
plt.show()

rf_importance = final_rf_fit.feature_importances_
df_importance = pd.DataFrame({'variable': X_train.columns, 'importance': rf_importance}).\
    sort_values('importance')

plt.barh(np.arange(len(df_importance.variable)), df_importance.importance)
plt.yticks(np.arange(len(df_importance.variable)), df_importance.variable)
plt.subplots_adjust(left=0.25)
plt.title("Feature Importance")
plt.show()
