import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ============================
# USER CONFIGURABLE PARAMETERS
# ============================
SYNTHETIC_CSV_PATH = "Data/Random_Resumes.csv"  # Update this path to where your synthetic data is saved
MAX_EXPERIENCE = 15  # Must match the value used during synthetic data generation

# ============================
# LOAD SYNTHETIC DATA
# ============================
df = pd.read_csv(SYNTHETIC_CSV_PATH)

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
# The synthetic CSV has "skills_similarity", "education_match", and "experience_diff".
# We'll create an "experience_closeness" feature so that higher values indicate a closer match.
df['experience_closeness'] = 1 - (df['experience_diff'] / MAX_EXPERIENCE)
# Now our features will be:
#   - skills_similarity: [0, 1] where 1 is a perfect match (from BERT-based similarity)
#   - education_match: binary (1 if education meets/exceeds, else 0)
#   - experience_closeness: [0, 1] where 1 means exact match in experience.

# Define the feature matrix X and target vector y.
features = ['skills_similarity', 'education_match', 'experience_closeness']
X = df[features]
y = df['match_score']

# ----------------------------
# SPLIT DATA INTO TRAIN AND TEST SETS
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# TRAIN THE LINEAR REGRESSION MODEL
# ----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# MAKE PREDICTIONS AND EVALUATE
# ----------------------------
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("=== Linear Regression Model ===")
print("Intercept:", model.intercept_)
print("Coefficients (for features {}):".format(features), model.coef_)

print("\n=== Training Set Performance ===")
print("Mean Squared Error (MSE):", train_mse)
print("R^2 Score:", train_r2)

print("\n=== Test Set Performance ===")
print("Mean Squared Error (MSE):", test_mse)
print("R^2 Score:", test_r2)
