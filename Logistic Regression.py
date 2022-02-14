import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns  
import shap
from sklearn import metrics
from numpy import interp
from sklearn.impute import SimpleImputer, KNNImputer
from matplotlib import pyplot
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, make_scorer, precision_score, brier_score_loss,roc_curve, roc_auc_score, auc, classification_report, precision_recall_curve, f1_score
from imblearn import under_sampling, over_sampling, combine
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold,cross_validate, KFold
from sklearn.calibration import calibration_curve, CalibratedClassifierCV


Data = pd.read_csv('D:/..../D1~D7 Dataset.csv', encoding='utf_8_sig')

train_data, test_data = train_test_split(Data, random_state=5, train_size=0.8)

Features = ['D1_FIO2', 'D2_FIO2', 'D3_FIO2', 'D4_FIO2', 'D5_FIO2', 'D6_FIO2', 'D7_FIO2',
            'D1_PEEPCPAP', 'D2_PEEPCPAP', 'D3_PEEPCPAP', 'D4_PEEPCPAP', 'D5_PEEPCPAP', 'D6_PEEPCPAP', 'D7_PEEPCPAP',
            'D1_PAW', 'D2_PAW', 'D3_PAW', 'D4_PAW', 'D5_PAW', 'D6_PAW', 'D7_PAW',
            'D1_MAPS', 'D2_MAPS', 'D3_MAPS', 'D4_MAPS', 'D5_MAPS', 'D6_MAPS', 'D7_MAPS',
            'D1_TOTRR', 'D2_TOTRR', 'D3_TOTRR', 'D4_TOTRR', 'D5_TOTRR', 'D6_TOTRR', 'D7_TOTRR',
            'NEW_D1_VTEXH', 'NEW_D2_VTEXH', 'NEW_D3_VTEXH', 'NEW_D4_VTEXH', 'NEW_D5_VTEXH', 'NEW_D6_VTEXH', 'NEW_D7_VTEXH',
            'D1_MVEXH', 'D2_MVEXH', 'D3_MVEXH', 'D4_MVEXH', 'D5_MVEXH', 'D6_MVEXH', 'D7_MVEXH',
            'D1_DDP', 'D2_DDP', 'D3_DDP', 'D4_DDP', 'D5_DDP', 'D6_DDP', 'D7_DDP', 'D2-D1_FIO2', 'D3-D2_FIO2', 'D4-D3_FIO2', 'D5-D4_FIO2', 'D6-D5_FIO2', 'D7-D6_FIO2',
            'D2-D1_PEEPCPAP', 'D3-D2_PEEPCPAP', 'D4-D3_PEEPCPAP', 'D5-D4_PEEPCPAP', 'D6-D5_PEEPCPAP', 'D7-D6_PEEPCPAP', 'D2-D1_PAW', 'D3-D2_PAW', 'D4-D3_PAW', 'D5-D4_PAW', 'D6-D5_PAW', 'D7-D6_PAW',
            'D1_temperature', 'D2_temperature', 'D3_temperature', 'D4_temperature', 'D5_temperature', 'D6_temperature', 'D7_temperature',
            'D1_Systolic', 'D1_Diastolic', 'D2_Systolic', 'D2_Diastolic', 'D3_Systolic', 'D3_Diastolic',
            'D4_Systolic', 'D4_Diastolic', 'D5_Systolic', 'D5_Diastolic', 'D6_Systolic', 'D6_Diastolic', 'D7_Systolic', 'D7_Diastolic',
            'D1_breath', 'D2_breath', 'D3_breath', 'D4_breath', 'D5_breath', 'D6_breath', 'D7_breath',
            'D1_pulse', 'D2_pulse', 'D3_pulse', 'D4_pulse', 'D5_pulse', 'D6_pulse', 'D7_pulse',
            'D1_SPO2', 'D2_SPO2', 'D3_SPO2', 'D4_SPO2', 'D5_SPO2', 'D6_SPO2', 'D7_SPO2',
            'ALB', 'BILT', 'ALKP', 'WBC', 'HGB', 'PLT', 'NEUT', 'CL', 'BUN', 'CREAT', 'LACTATE', 'PTP', 'PH_A', 'PO2_A', 'HCO3_A',
            'D1_Drainage', 'D2_Drainage', 'D3_Drainage', 'D4_Drainage', 'D5_Drainage', 'D6_Drainage', 'D7_Drainage',
            'D1_Urine', 'D2_Urine', 'D3_Urine', 'D4_Urine', 'D5_Urine', 'D6_Urine', 'D7_Urine',
            'D1_Injection', 'D2_Injection', 'D3_Injection', 'D4_Injection', 'D5_Injection', 'D6_Injection', 'D7_Injection',
            'D1_Diet', 'D2_Diet', 'D3_Diet', 'D4_Diet', 'D5_Diet', 'D6_Diet', 'D7_Diet', 'Apache']



new_pd_train = pd.DataFrame(train_data[Features])
X_train = new_pd_train.to_numpy()

# Imputation of missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer = KNNImputer(n_neighbors=20)
imp = imputer.fit(X_train)   # 對數據集進行分析擬合
X_train = imputer.transform(X_train)  # 對數據集進行變換
Y_train = train_data["die(365)"]
# Pandas Series 轉為 DataFrame
X_train = pd.DataFrame(data=X_train, columns=Features)       


# 處理不平衡資料
new_pd_test = pd.DataFrame(test_data[Features])
X_test = new_pd_test.to_numpy()
imp = imputer.fit(X_test)   # 對數據集進行分析擬合
X_test = imputer.transform(X_test)  # 對數據集進行變換
Y_test = test_data["die(365)"]
# Pandas Series 轉為 DataFrame
X_test = pd.DataFrame(data=X_test, columns=Features)


# estimate scale_pos_weight value
counter_train = Counter(Y_train)
estimate_train = counter_train[0] / counter_train[1]
estimate_train

def classification_report_with_accuracy_score(y_true, y_pred):
    print (classification_report(y_true, y_pred,digits=3)) # print classification report
    return accuracy_score(y_true, y_pred) # return accuracy score
    
# CV model
scoring = {'report':    make_scorer(classification_report_with_accuracy_score),
           'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'sensitivity'  : make_scorer(recall_score),
           'specificity': make_scorer(recall_score,pos_label=0),
           'F-1': make_scorer(f1_score),
           'auc': make_scorer(roc_auc_score, needs_proba=True)
          }
          
LR_model = LogisticRegression(solver='liblinear', max_iter=10000, class_weight="balanced")

#kfold = StratifiedKFold(n_splits=5, random_state=None)
kfold = KFold(n_splits=5, random_state=0, shuffle=True)
results = cross_validate(LR_model, X_train, Y_train, cv=kfold,scoring=scoring)

LR_model.fit(X_train, Y_train)
LR_y_pred_test = LR_model.predict(X_test)
LR_y_preds_proba_test = LR_model.predict_proba(X_test)

# # #計算auc
LR_auc_test = roc_auc_score(Y_test, LR_y_preds_proba_test[:, 1])
LR_fpr_test, LR_tpr_test, LR_thresholds_test = roc_curve(Y_test, LR_y_preds_proba_test[:, 1])


# performance
print('Model: Logistic Regression\n')
print(classification_report(Y_test, LR_y_pred_test,
      target_names=['Survive (Class 0)', 'Death (Class 1)'],digits=3))
print(f'Accuracy Score: {accuracy_score(Y_test,LR_y_pred_test)}')
print(f'Confusion Matrix: \n{confusion_matrix(Y_test, LR_y_pred_test)}')
print(f'Area Under Curve: {roc_auc_score(Y_test, LR_y_preds_proba_test[:, 1])}')
print(f'Recall score: {recall_score(Y_test,LR_y_pred_test)}')
print(f'Brier score: {brier_score_loss(Y_test, LR_y_preds_proba_test[:, 1])}')
print("###########################################################\n")


# ROC CURVE
plt.figure()
plt.figure(figsize=(12, 12))
plt.plot(LR_fpr_test, LR_tpr_test, 'black', label='LR (AUC = %0.3f)' % LR_auc_test, color='g',lw=6)
plt.plot([0, 1], [0, 1], color='black', lw=6, linestyle='--', alpha=.8)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
# plt.xlabel('False Positive Rate', fontsize=40)
# plt.ylabel('True Positive Rate', fontsize=40)
# plt.title('ROC CURVE (30 DAY)', fontsize=50)
plt.legend(loc="lower right",fontsize=30)
sns.set(style='white') # 白色網格背景
plt.rcParams["font.weight"] = "bold"




