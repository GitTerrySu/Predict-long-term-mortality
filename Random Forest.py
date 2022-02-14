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
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
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
  
  
BRF_model =BalancedRandomForestClassifier(n_estimators=3000, criterion='entropy', verbose=1)

kfold = StratifiedKFold(n_splits=5, random_state=None)
results = cross_validate(BRF_model, X_train, y_train, cv=kfold,scoring=scoring)
print("Accuracy: %.2f%% (%.2f%%)" % (results['test_accuracy'].mean()*100, results['test_accuracy'].std()*100))
print("Precision: %.2f%% (%.2f%%)" % (results['test_precision'].mean()*100, results['test_precision'].std()*100))
print("Sensitivity: %.2f%% (%.2f%%)" % (results['test_sensitivity'].mean()*100, results['test_sensitivity'].std()*100))
print("Specificity: %.2f%% (%.2f%%)" % (results['test_specificity'].mean()*100, results['test_specificity'].std()*100))
print("AUC: %.2f%% (%.2f%%)" % (results['test_auc'].mean()*100, results['test_auc'].std()*100))


############################################################################################################################
BRF_model.fit(X_train, y_train)
BRF_y_pred_test = BRF_model.predict(X_test)
BRF_y_preds_proba_test = BRF_model.predict_proba(X_test)

# #計算auc
BRF_auc_test = roc_auc_score(y_test, BRF_y_preds_proba_test[:, 1])
BRF_fpr_test, BRF_tpr_test, BRF_thresholds_test = roc_curve(y_test, BRF_y_preds_proba_test[:, 1])



# performance
print('Model: Balanced Random Forest\n')
print(classification_report(y_test, BRF_y_pred_test, target_names=['Not Ready (Class 0)', 'Ready (Class 1)'],digits=3))
print(f'Accuracy Score: {accuracy_score(y_test,BRF_y_pred_test)}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, BRF_y_pred_test)}')
print(f'Area Under Curve: {roc_auc_score(y_test, BRF_y_preds_proba_test[:,1])}')
print(f'Recall score: {recall_score(y_test,BRF_y_pred_test)}')
print(f'Brier score: {brier_score_loss(y_test, BRF_y_preds_proba_test[:,1])}')
print("###########################################################\n")

# ROC CURVE
plt.figure()
plt.figure(figsize=(12, 12))
plt.plot(BRF_fpr_test, BRF_tpr_test, 'black', label='RF (AUC = %0.3f)' %BRF_auc_test, color='b',lw=6)
plt.plot([0, 1], [0, 1], color='black', lw=6, linestyle='--', alpha=.8)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
# plt.xlabel('False Positive Rate', fontsize=40)
# plt.ylabel('True Positive Rate', fontsize=40)
# plt.title('ROC CURVE (30 DAY)', fontsize=50)
plt.legend(loc="lower right",fontsize=30)
sns.set(style='white') 
plt.rcParams["font.weight"] = "bold"
# sns.despine(top=True, right= True) 
# plt.grid(False)
# plt.show() 
#plt.savefig('ROC(Testing,365).tif', format='tif', dpi=300, bbox_inches='tight')
