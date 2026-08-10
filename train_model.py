"""
Network Attack Classification System
Dataset: NSL-KDD (KDDTrain+ / KDDTest+)
Models: Decision Tree, Random Forest, SVM
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import json

COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

# Map specific attack names to 5 broad categories (standard NSL-KDD grouping)
ATTACK_MAP = {
    'normal': 'normal',
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos', 'smurf': 'dos',
    'teardrop': 'dos', 'apache2': 'dos', 'udpstorm': 'dos', 'processtable': 'dos', 'worm': 'dos',
    'satan': 'probe', 'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe',
    'mscan': 'probe', 'saint': 'probe',
    'guess_passwd': 'r2l', 'ftp_write': 'r2l', 'imap': 'r2l', 'phf': 'r2l', 'multihop': 'r2l',
    'warezmaster': 'r2l', 'warezclient': 'r2l', 'spy': 'r2l', 'xlock': 'r2l', 'xsnoop': 'r2l',
    'snmpguess': 'r2l', 'snmpgetattack': 'r2l', 'httptunnel': 'r2l', 'sendmail': 'r2l', 'named': 'r2l',
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'rootkit': 'u2r', 'perl': 'u2r',
    'sqlattack': 'u2r', 'xterm': 'u2r', 'ps': 'u2r'
}

def load_data(path):
    df = pd.read_csv(path, names=COLUMNS)
    df = df.drop('difficulty', axis=1)
    df['attack_category'] = df['label'].map(lambda x: ATTACK_MAP.get(x, 'r2l'))
    return df

print("Loading NSL-KDD dataset...")
train_df = load_data('data/KDDTrain+.txt')
test_df = load_data('data/KDDTest+.txt')
print(f"Train: {train_df.shape}, Test: {test_df.shape}")
print(train_df['attack_category'].value_counts())

# Encode categorical features
cat_cols = ['protocol_type', 'service', 'flag']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]], axis=0)
    le.fit(combined)
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    encoders[col] = le

# Encode target
target_le = LabelEncoder()
target_le.fit(train_df['attack_category'])
y_train = target_le.transform(train_df['attack_category'])
y_test = target_le.transform(test_df['attack_category'])

feature_cols = [c for c in train_df.columns if c not in ['label', 'attack_category']]
X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

def evaluate(name, model, X_te, y_te):
    preds = model.predict(X_te)
    acc = accuracy_score(y_te, preds)
    prec = precision_score(y_te, preds, average='weighted', zero_division=0)
    rec = recall_score(y_te, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_te, preds, average='weighted', zero_division=0)
    print(f"\n{name}: Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")
    print(classification_report(y_te, preds, target_names=target_le.classes_, zero_division=0))
    results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    return preds

print("\n=== Training Decision Tree ===")
dt = DecisionTreeClassifier(max_depth=15, random_state=42)
dt.fit(X_train, y_train)
evaluate('Decision Tree', dt, X_test, y_test)

print("\n=== Training Random Forest ===")
rf = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
evaluate('Random Forest', rf, X_test, y_test)

print("\n=== Training SVM (on subsample for speed) ===")
# SVM is slow on 125k rows; use a stratified subsample for training as is standard practice
from sklearn.model_selection import train_test_split
X_svm, _, y_svm, _ = train_test_split(X_train_scaled, y_train, train_size=15000, stratify=y_train, random_state=42)
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_svm, y_svm)
evaluate('SVM', svm, X_test_scaled, y_test)

# Save best model (Random Forest typically wins) + all artifacts
joblib.dump(rf, 'models/random_forest.pkl')
joblib.dump(dt, 'models/decision_tree.pkl')
joblib.dump(svm, 'models/svm.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(encoders, 'models/encoders.pkl')
joblib.dump(target_le, 'models/target_encoder.pkl')
joblib.dump(feature_cols, 'models/feature_cols.pkl')

with open('models/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n=== Final Results Summary ===")
for name, r in results.items():
    print(f"{name}: {r}")
print("\nAll models saved to models/")

# ---- Additional: standard train/test split evaluation (same-distribution) ----
# This matches the ~97-99% accuracy commonly reported in NSL-KDD papers,
# since it evaluates on unseen data drawn from the SAME distribution as training
# (unlike KDDTest+ which deliberately includes novel attack types).
print("\n\n=== Same-distribution split evaluation (train_test_split on KDDTrain+) ===")
from sklearn.model_selection import train_test_split
Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

rf2 = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1)
rf2.fit(Xs_tr, ys_tr)
preds2 = rf2.predict(Xs_te)
acc2 = accuracy_score(ys_te, preds2)
f1_2 = f1_score(ys_te, preds2, average='weighted', zero_division=0)
print(f"Random Forest (same-distribution split): Accuracy={acc2:.4f}, F1={f1_2:.4f}")

dt2 = DecisionTreeClassifier(max_depth=15, random_state=42)
dt2.fit(Xs_tr, ys_tr)
preds_dt2 = dt2.predict(Xs_te)
acc_dt2 = accuracy_score(ys_te, preds_dt2)
print(f"Decision Tree (same-distribution split): Accuracy={acc_dt2:.4f}")
