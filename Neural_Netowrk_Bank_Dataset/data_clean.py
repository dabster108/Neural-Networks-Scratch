# data_clean.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Load dataset
df = pd.read_csv("Neural_Netowrk_Bank_Dataset/bank.csv")

# Check dataset info
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values per column:")
print(df.isnull().sum())

# Step 2: Convert target 'deposit' (yes/no -> 1/0)
df['deposit'] = df['deposit'].map({'yes': 1, 'no': 0})

# Step 3: Drop duplicates (if any)
dups = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {dups}")
if dups > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Dropped duplicates. New shape:", df.shape)

# Step 4: Encode categorical variables (one-hot encoding)
df_encoded = pd.get_dummies(df, columns=['job','marital','education','default',
                                         'housing','loan','contact','month','poutcome'],
                            drop_first=True)

print("Shape after encoding:", df_encoded.shape)

# Step 5: Separate features and target
X = df_encoded.drop('deposit', axis=1)
y = df_encoded['deposit']

# Step 6: Normalize numeric columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Final feature matrix shape:", X_scaled.shape)

# Step 7: Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Expose variables for import