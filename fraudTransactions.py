# PACKAGES
import pandas as pd
import numpy as np
import holidays

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc, recall_score
from sklearn.utils import class_weight
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier


import tensorflow as tf
from tensorflow.keras.metrics import AUC
from tensorflow.keras import backend as K
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import sys
import os
import warnings
warnings.filterwarnings("ignore")  # Hide all warnings
sys.stdout = open(os.devnull, 'w')
sys.stdout = sys.__stdout__
###

# AUXILIARY FUNCTIONS
def cumulative_sum(df, row_idx, time, column='transactionAmount'):
    end_index = row_idx
    start_index = row_idx
    account = df.loc[row_idx, 'accountNumber']
    end_time = df.loc[row_idx, 'transactionTime']
    while start_index>0:
        if df.loc[start_index-1, 'accountNumber'] != account:
            break
        else:
            if (end_time -df.loc[start_index-1, 'transactionTime']).total_seconds() < time:
                start_index -= 1
            else:
                break
    if start_index == end_index:
        return df.loc[row_idx, column]
    return df.loc[start_index:end_index, column].sum()


def compute_merchant_risk_score(df):
    """
    Computes a fraud risk score for each merchant based on historical fraud rates.
    
    Parameters:
    df (pd.DataFrame): Transaction data with 'merchantId' and 'flag' columns.
    
    Returns:
    pd.Series: A mapping of merchants to their fraud risk score.
    """
    # Compute fraud rate per merchant
    merchant_fraud_rates = df.groupby('merchantId')['flag'].mean()
    
    # Normalize scores between 0 and 1
    risk_score = (merchant_fraud_rates - merchant_fraud_rates.min()) / (merchant_fraud_rates.max() - merchant_fraud_rates.min())
    
    return risk_score

def has_transacted_before(df):
    """
    Checks if a customer has transacted with a merchant before.
    
    Parameters:
    df (pd.DataFrame): Transaction data with 'accountNumber' and 'merchantId'.
    
    Returns:
    pd.Series: 1 if the customer has transacted before, 0 otherwise.
    """
    seen = set()
    history = []

    for _, row in df.iterrows():
        key = (row['accountNumber'], row['merchantId'])
        if key in seen:
            history.append(1)  # Has transacted before
        else:
            history.append(0)  # First-time transaction
            seen.add(key)

    return pd.Series(history)

def geographic_deviation(df):
    """
    Flags transactions where the merchant country is unusual for the user.
    
    Parameters:
    df (pd.DataFrame): Transaction data with 'accountNumber' and 'merchantCountry'.
    
    Returns:
    pd.Series: 1 if the country is unusual, 0 otherwise.
    """
    user_country_map = df.groupby('accountNumber')['merchantCountry'].agg(lambda x: x.mode()[0])
    return (df['merchantCountry'] != df['accountNumber'].map(user_country_map)).astype(int)

def merchant_zip_consistency(df, feature = 'merchantZip'):
    """
    Flags if a customer's transaction is in an unusual ZIP code area based on past transactions.
    
    Parameters:
    df (pd.DataFrame): Transaction data with 'accountNumber' and 'merchantZip' columns.
    
    Returns:
    pd.Series: 1 if the transaction is in an unusual ZIP code, 0 otherwise.
    """
    # Find the most frequent ZIP code for each customer
    user_zip_map = df.groupby('accountNumber')['merchantZip'].agg(lambda x: x.mode()[0])
    
    # Compare current transaction ZIP with the most frequent ZIP for that customer
    return (df['merchantZip'] != df['accountNumber'].map(user_zip_map)).astype(int)

def compute_country_risk_score(df):
    """
    Computes a fraud risk score for each country based on historical fraud rates.
    
    Parameters:
    df (pd.DataFrame): Transaction data with 'merchantCountry' and 'isFraud' columns.
    
    Returns:
    pd.Series: A mapping of countries to their fraud risk score.
    """
    country_fraud_rates = df.groupby('merchantCountry')['flag'].mean()
    # Normalize fraud rates to get a risk score between 0 and 1
    risk_score = (country_fraud_rates - country_fraud_rates.min()) / (country_fraud_rates.max() - country_fraud_rates.min())
    return risk_score
######


### IMPORTANT TO HAVE THESE FILES IN THE FOLDER DATA!!!
# We read the data provided
df = pd.read_csv("Data/transactions_obf.csv")
aux = pd.read_csv("Data/labels_obf.csv")

# We create the target variable 'flag'
df = df.merge(aux, how='left', on=['eventId'])
df['flag'] = np.where(df.reportedTime.notnull(), 1, 0)


df['transactionTime']= pd.to_datetime(df['transactionTime'])
df['reportedTime']= pd.to_datetime(df['reportedTime'])
df['daysDetection'] = (df['reportedTime'] - df['transactionTime']).dt.total_seconds() / (3600*24)

df = df.sort_values(by=['accountNumber', 'transactionTime'])
df = df.reset_index()

# df['daysDetection'].describe() # We can discover the average time to flag (9.5 days)


# FEATURE ENGINEERING

    # TRANSACTION TIME: HOUR, MONTH, YEAR, DAYTYPE, HOLIDAY, ORDERTRANSACTION (First, second,... of the hour, day), DELTAPREVTRANS, SAMEPAYMENT
df['hour'] = df['transactionTime'].dt.hour
df['month'] = df['transactionTime'].dt.month
df['year'] = df['transactionTime'].dt.year
df['dayType'] = df['transactionTime'].dt.dayofweek  # Monday=0, Sunday=6
uk_holidays = holidays.UK(years=[2017,2018])  # You can change this to other countries: holidays.US(), but we know it is mostly UK for the ZIP code
df['isHoliday'] = df['transactionTime'].dt.date.isin(uk_holidays)
df['orderDay'] = df.groupby(['accountNumber', df['transactionTime'].dt.date]).cumcount() + 1
df['orderWeek'] = df.groupby(['accountNumber', df['transactionTime'].dt.to_period('W')]).cumcount() + 1
df['orderMonth'] = df.groupby(['accountNumber', df['transactionTime'].dt.to_period('M')]).cumcount() + 1
df['timeDiffHours'] = df.groupby('accountNumber')['transactionTime'].diff().dt.total_seconds() / 3600 # Measure in hours
df['timeDiffHours'] = df['timeDiffHours'].fillna(-1)
df['transactionDate'] = df['transactionTime'].dt.date
df['sameDayPayment'] = df.groupby(['accountNumber', 'merchantId', 'transactionDate', 'transactionAmount'])['eventId'].transform('count') > 1

    # ACCOUNTNUMBER: QUARTILE OF MOST FREQUENT ACCOUNTS
freq = df['accountNumber'].value_counts()
ranking = {cat: rank + 1 for rank, (cat, _) in enumerate(freq.items())}
df['ranking'] = df['accountNumber'].map(ranking)
quartiles = df['ranking'].quantile([0.25, 0.5, 0.75]).values
bins = [df['ranking'].min() - 1] + list(quartiles) + [df['ranking'].max() + 1]
df['quartileNumberTransactions'] = pd.cut(df['ranking'], bins=bins, labels=['1', '2', '3', '4']).astype(int)

    # TRANSACTION AMOUNT: CUMULATIVE SUM, RATIO, DIFFERENCE FROM AVERAGE, PERCENTILEHIGHEST
if False: # We avoid this because it takes some minutes and the efficiency of the model did not improve.
    df['amountLastHour'] = df.index.to_series().apply(lambda idx: cumulative_sum(df, idx, 3600))
    df['amountLastDay'] = df.index.to_series().apply(lambda idx: cumulative_sum(df, idx, 3600*24))
    df['amountLastWeek'] = df.index.to_series().apply(lambda idx: cumulative_sum(df, idx, 3600*24*7))
    df['amountLastMonth'] = df.index.to_series().apply(lambda idx: cumulative_sum(df, idx, 3600*24*30))
df['avgTransactionAmount'] = df.groupby('accountNumber')['transactionAmount'].transform('mean')
df['transactionRatio'] = df['transactionAmount'] / (df['availableCash']+1)
df['spendingStd'] = df.groupby('accountNumber')['transactionAmount'].transform('std')
df['spendingStd'] = df['spendingStd'].fillna(0)
df['diffFromAvg'] = df['transactionAmount'] - df['avgTransactionAmount']
df['diffStd'] = df.apply(lambda row: row['diffFromAvg'] / row['spendingStd'] if row['spendingStd'] != 0 else 0, axis=1)
df['isHighAmount'] = df['transactionAmount'] > df['transactionAmount'].quantile(0.90)
df['isHighAmount2'] = df['transactionAmount'] > df['transactionAmount'].quantile(0.95)
df['isHighAmount3'] = df['transactionAmount'] > df['transactionAmount'].quantile(0.99)

    # MERCHANT: RISKSCORE (most important), FREQUENTMERCHANT
df['merchantRiskScore'] = df['merchantId'].map(compute_merchant_risk_score(df[df.year == 2017]))
df['merchantRiskScore'] = df['merchantRiskScore'].fillna(1)
df['frequentMerchant'] = has_transacted_before(df)
df['mcc_group'] = df['mcc'].astype(str).str[:2].astype(int) # We can group the 4 digit code, with only the first two REF: https://usa.visa.com/content/dam/VCOM/download/merchants/visa-merchant-data-standards-manual.pdf
#TBD if it is the first time the merchant appears in the historical data...

    # ZONE: GEODEVIATION, ZONE/AREA-CONSISTENCY, COUNTRYRISK
df['geoDeviation'] = geographic_deviation(df)
df['merchantZip'] = df['merchantZip'].astype(str)
df['merchantZipLetters'] = df['merchantZip'].str.extract('([A-Za-z]+)', expand=False)
df['zipConsistency'] = merchant_zip_consistency(df)
df['areaConsistency'] = merchant_zip_consistency(df, feature = 'merchantZipLetters')
df['countryRiskScore'] = df['merchantCountry'].map(compute_country_risk_score(df[df.year == 2017]))
df['countryRiskScore'] = df['countryRiskScore'].fillna(1)


# ONE-HOT ENCODING
df2 = pd.get_dummies(df, columns=['posEntryMode', 'merchantCountry', 'merchantZipLetters'], dtype='int')




# PREPARE TRAINING AND TEST DATA
drop_columns = ['index', 'merchantId', 'transactionTime', 'transactionDate', 'accountNumber', 'merchantZip', 'eventId', 'mcc', 'flag', 'reportedTime', 'daysDetection', 'ranking', 'index']
X_train = df2[df2.year == 2017].drop(columns=drop_columns, errors = 'ignore')
X_test = df2[df2.year == 2018].drop(columns=drop_columns, errors = 'ignore')
y_train = df2[df2.year == 2017]['flag']
y_test = df2[df2.year == 2018]['flag']

# SCALING AMOUNTS
scaler = StandardScaler()
X_train[['transactionAmount', 'availableCash']] = scaler.fit_transform(X_train[['transactionAmount', 'availableCash']])
X_test[['transactionAmount', 'availableCash']] = scaler.transform(X_test[['transactionAmount', 'availableCash']])


# LOGISTIC REGRESSION: FEATURE IMPORTANCE
model = LogisticRegression(solver='liblinear', class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Logistic Regression: Model Performance:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("----------------------")
coefficients = model.coef_[0]  # Coefficients of features for binary classification
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': coefficients
})
feature_importance['Importance'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print(feature_importance)


# OTHER MODELS
models = {
    'RandomForest': RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample', max_depth=10, random_state=42),
    #'SVC': SVC(kernel='rbf', C=1, gamma='scale', random_state=42), #SLOW AND NOT GOOD
    'XGBoost': XGBClassifier(eval_metric='logloss', scale_pos_weight=y_train.value_counts()[0] / y_train.value_counts()[1], max_depth=7, learning_rate=0.1, n_estimators=300, random_state=42)#,
    # ANOMALIES
    #'OneSVM':OneClassSVM(nu=0.01, kernel='rbf', gamma='scale'), # SLOW
    #'IsoForest': IsolationForest(contamination=0.01, random_state=42)

}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Model Performance:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("----------------------")


# DEEP LEARNING
# Step 1: Define a custom callback to compute F1 during training
class F1ScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_train, y_train, X_val, y_val, top_train=4800, top_val=400):
        super(F1ScoreCallback, self).__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.top_train = top_train
        self.top_val = top_val

    def on_epoch_end(self, epoch, logs=None):
        # Get predicted probabilities for training and validation sets
        train_probabilities = self.model.predict(self.X_train)
        val_probabilities = self.model.predict(self.X_val)
        
        # Get the indices of the top 4800 (train) and 400 (val) highest probabilities
        top_train_idx = np.argsort(train_probabilities.flatten())[::-1][:self.top_train]
        top_val_idx = np.argsort(val_probabilities.flatten())[::-1][:self.top_val]
        
        # Create adjusted labels (1 for the top probabilities, else 0)
        train_predicted_labels = np.zeros_like(self.y_train)
        train_predicted_labels[top_train_idx] = 1
        
        val_predicted_labels = np.zeros_like(self.y_val)
        val_predicted_labels[top_val_idx] = 1
        
        # Compute F1 score for both train and validation sets
        train_f1 = f1_score(self.y_train, train_predicted_labels)
        val_f1 = f1_score(self.y_val, val_predicted_labels)
        
        # Print F1 scores for both train and validation sets
        print(f'Epoch {epoch+1} - Train F1: {train_f1:.4f} - Validation F1: {val_f1:.4f}')

def f1_score(y_true, y_pred):
    # Ensure both tensors are the same type
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    # Convert probabilities to binary (0 or 1)
    y_pred = K.round(y_pred)  
    
    tp = K.sum(y_true * y_pred)  # True Positives
    fp = K.sum((1 - y_true) * y_pred)  # False Positives
    fn = K.sum(y_true * (1 - y_pred))  # False Negatives
    
    precision = tp / (tp + fp + K.epsilon())  # Precision calculation
    recall = tp / (tp + fn + K.epsilon())  # Recall calculation
    
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())  # F1 score calculation
    return f1

def f1_loss_class1(y_true, y_pred):
    """
    Custom loss function to maximize F1-score for class 1.
    """
    y_true = K.cast(y_true, 'float32')
    y_pred = K.clip(y_pred, 1e-7, 1 - 1e-7)  # Prevent log(0) issues

    tp = K.sum(y_true * y_pred)  # True Positives
    fp = K.sum((1 - y_true) * y_pred)  # False Positives
    fn = K.sum(y_true * (1 - y_pred))  # False Negatives

    precision = tp / (tp + fp + K.epsilon())  # Precision
    recall = tp / (tp + fn + K.epsilon())  # Recall

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())  # F1-score
    return 1 - f1  # Subtract from 1 to make it a loss function

# Define Focal Loss
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, 'float32')
        y_pred = K.clip(y_pred, 1e-7, 1 - 1e-7)  # Prevent log(0) issues
        
        # Compute focal loss
        loss = -y_true * alpha * K.pow(1 - y_pred, gamma) * K.log(y_pred) \
               - (1 - y_true) * (1 - alpha) * K.pow(y_pred, gamma) * K.log(1 - y_pred)
        return K.mean(loss)
    
    return loss

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
f1_callback = F1ScoreCallback(X_train_scaled, y_train, X_test, y_test)

# 1. Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Apply SMOTE to balance the data by oversampling the minority class
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
X_train_resampled = X_train_scaled
y_train_resampled = y_train

# 3. Compute class weights to handle imbalance
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# 4. Build the Neural Network Model
nn_model = Sequential([
    Dense(64, input_dim=X_train_resampled.shape[1], activation='relu'),
    Dropout(0.5),  # Dropout for regularization
    Dense(32, activation='relu'),
    Dropout(0.5),  # Dropout for regularization
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification (fraud vs non-fraud)
])

# Compile the model (several options)
#nn_model.compile(optimizer='adam', loss=f1_loss_class1, metrics=['Recall', 'AUC'])
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['Recall'])

# 5. Set up callbacks for early stopping and reducing learning rate on plateau
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# 6. Train the model with class weights and callbacks
nn_model.fit(X_train_resampled, y_train_resampled, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test),
             class_weight=class_weight_dict, callbacks=[early_stopping, reduce_lr, f1_callback])

# 7. Predict using the trained model
nn_predictions = nn_model.predict(X_test_scaled)
nn_predictions = (nn_predictions > 0.5).astype(int)  # Convert probabilities to binary labels

# 8. Evaluate the model
print("Classification Report:")
print(classification_report(y_test, nn_predictions))

# 9. Compute ROC-AUC
roc_auc = roc_auc_score(y_test, nn_predictions)
print(f'ROC-AUC: {roc_auc}')

# 10. Compute Precision-Recall AUC
precision, recall, _ = precision_recall_curve(y_test, nn_model.predict(X_test_scaled))
pr_auc = auc(recall, precision)
print(f'PR-AUC: {pr_auc}')


fraud_probabilities = nn_model.predict(X_test_scaled)  # Predict probabilities for fraud (class 1)

# 11: RECALL OF THE TEST WITH 400 PREDICTIONS
top_400_idx = np.argsort(fraud_probabilities.flatten())[::-1][:400]  # Sort in descending order and pick top 400
predicted_labels = np.zeros_like(fraud_probabilities.flatten()) # Create a binary array for the entire test set, initially set to 0
predicted_labels[top_400_idx] = 1 #Assign 1 to the top 400 predictions with the highest likelihood of fraud
recall_all = recall_score(y_test, predicted_labels) # Compute Recall for the whole test dataset
print(f'Recall for the whole test set (considering only top 400 fraud predictions): {recall_all:.4f}')