
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from scipy.special import expit as logistic
from scipy.linalg import LinAlgError

with open("heart_transplant.txt", 'r') as f:
    lines = f.readlines()
parsed_data = [list(map(int, line.strip().split())) for line in lines]

column_names = ['Age_at_Transplant', 'Survival_Status', 'Survival_Time']
heart_data = pd.DataFrame(parsed_data, columns=column_names)

heart_data.head()

data_info = heart_data.info()

data_description = heart_data.describe()
round(data_description,2)

plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
sns.histplot(heart_data['Age_at_Transplant'], kde=True, bins=20)
plt.title('Distribution of Age at Transplant')
plt.xlabel('Age at Transplant')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
ax = sns.countplot(x='Survival_Status', data=heart_data)
plt.title('Distribution of Survival Status')
plt.xlabel('Survival Status')
plt.ylabel('Frequency (%)')
total = len(heart_data['Survival_Status'])
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')
plt.xticks([0, 1], ['Didn\'t Survive', 'Survived'])

plt.subplot(1, 3, 3)
sns.histplot(heart_data['Survival_Time'], kde=True, bins=20)
plt.title('Distribution of Survival Time')
plt.xlabel('Survival Time (days)')
plt.ylabel('Frequency')

X = heart_data[['Age_at_Transplant']]
y = heart_data['Survival_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age_at_Transplant', y='Survival_Status', data=heart_data, hue='Survival_Status', palette="viridis")
plt.title('Logistic Regression Decision Boundary')

x_values = np.linspace(20, 65, 300)
x_values = x_values.reshape(-1, 1)
y_values = model.predict_proba(x_values)[:, 1]

plt.plot(x_values, y_values, label='Decision Boundary', color='red')
plt.xlabel('Age_at_Transplant')
plt.ylabel('Probability of Survival')
plt.legend()
plt.show()

print(f"Model Accuracy: {accuracy:.2f}")

def logistic(Y):
    return 1 / (1 + np.exp(-Y))

def heaviside(Y):
    return np.heaviside(Y, 0.5)

plt.figure(figsize=(12, 8))
sns.scatterplot(x='Age_at_Transplant', y='Survival_Status', data=heart_data, hue='Survival_Status', palette="viridis")
plt.title('Logistic Regression vs. Heaviside Function')

x_values = np.linspace(20, 65, 300)
x_values_reshaped = x_values.reshape(-1, 1)
y_prob_values = model.predict_proba(x_values_reshaped)[:, 1]
y_heaviside_values = heaviside(y_prob_values - 0.5)

plt.plot(x_values, y_prob_values, label='Logistic Function (Soft)', color='red', linestyle='--')
plt.plot(x_values, y_heaviside_values, label='Heaviside Function (Hard)', color='blue', linestyle='-.')
plt.xlabel('Age_at_Transplant')
plt.ylabel('Function Value')
plt.legend()
plt.show()


conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

class_report = classification_report(y_test, y_pred)
print(class_report)

plt.scatter(X, y, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Age at Transplant')
plt.ylabel('Survival Status')
plt.title('Logistic Regression Model Fit')
plt.legend()

heart_failure_data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

heart_failure_data.info()
heart_failure_data.head()

X_multi = heart_failure_data.drop(['DEATH_EVENT'], axis=1)
y_multi = heart_failure_data['DEATH_EVENT']

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

multi_model = LogisticRegression(max_iter=1000)

multi_model.fit(X_train_multi, y_train_multi)

y_pred_multi = multi_model.predict(X_test_multi)

multi_accuracy = accuracy_score(y_test_multi, y_pred_multi)
print(f'Accuracy: {multi_accuracy}')

X_mle = heart_failure_data[['ejection_fraction', 'serum_creatinine']].values
y_mle = heart_failure_data['DEATH_EVENT'].values
X_mle = np.hstack([np.ones((X_mle.shape[0], 1)), X_mle])
X_train, X_test, y_train, y_test = train_test_split(X_mle, y_mle, test_size=0.2, random_state=42)


def log_likelihood(theta, X, y):
    epsilon = 1e-5
    g = 1 / (1 + np.exp(-X.dot(theta)))
    return np.sum(y * np.log(g + epsilon) + (1 - y) * np.log(1 - g + epsilon))

mle_model = LogisticRegression(fit_intercept=False)
mle_model.fit(X_train, y_train)
theta_mle = mle_model.coef_.reshape(-1)
theta_mle = mle_model.coef_.reshape(-1)

log_likelihood_value = log_likelihood(theta_mle, X_test, y_test)
log_likelihood_value


features_all = ['age', 'anaemia', 'high_blood_pressure', 'creatinine_phosphokinase', 'diabetes', 
                'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
X_mle_all_features = heart_failure_data[features_all].values
y_mle_all_features = heart_failure_data['DEATH_EVENT'].values
X_mle_all_features = np.hstack([np.ones((X_mle_all_features.shape[0], 1)), X_mle_all_features])
X_train_all_features, X_test_all_features, y_train_all_features, y_test_all_features = train_test_split(X_mle_all_features, y_mle_all_features, test_size=0.2, random_state=42)

mle_model_all_features = LogisticRegression(fit_intercept=False, max_iter=1000)
mle_model_all_features.fit(X_train_all_features, y_train_all_features)
theta_mle_all_features = mle_model_all_features.coef_.reshape(-1)

log_likelihood_value_all_features = log_likelihood(theta_mle_all_features, X_test_all_features, y_test_all_features)
log_likelihood_value_all_features


def gradient_ascent_path(X, y, alpha=0.01, max_iter=100):
    """
    Run gradient ascent to maximize the log-likelihood for logistic regression.
    
    Parameters:
    X (array): The feature matrix.
    y (array): The labels vector.
    alpha (float): The learning rate.
    max_iter (int): The number of iterations to run the algorithm.
    
    Returns:
    array: The final theta values.
    array: The theta values at each iteration.
    """
    theta = np.zeros((X.shape[1], 1))
    all_thetas = np.zeros((X.shape[1], max_iter+1))
    all_thetas[:, 0:1] = theta
    for t in range(max_iter):
        G = 1 / (1 + np.exp(-X.dot(theta)))
        gradient = X.T.dot(y - G)
        gradient /= np.linalg.norm(gradient, ord=2)
        theta += alpha * gradient.reshape(-1, 1)
        all_thetas[:, t+1:t+2] = theta
    
    return theta, all_thetas

X_train_np = np.array(X_train)
y_train_reshaped = np.array(y_train).reshape(-1, 1)
final_theta, all_thetas = gradient_ascent_path(X_train_np, y_train_reshaped, alpha=0.1, max_iter=100)
final_theta, all_thetas[:,:5]

theta_0_vals = np.linspace(theta_mle[0] - 1, theta_mle[0] + 1, 100)
theta_1_vals = np.linspace(theta_mle[1] - 1, theta_mle[1] + 1, 100)
theta_0_grid, theta_1_grid = np.meshgrid(theta_0_vals, theta_1_vals)

ll_grid = np.zeros((100, 100))

for i in range(100):
    for j in range(100):
        theta_tmp = np.array([theta_0_grid[i, j], theta_1_grid[i, j], theta_mle[2]]) 
        ll_grid[i, j] = log_likelihood(theta_tmp, X_test, y_test)

plt.figure(figsize=(7, 7))
cp = plt.contourf(theta_0_grid, theta_1_grid, ll_grid, cmap='viridis')
plt.colorbar(cp)
plt.title('Log-likelihood Countour Plot with Gradient Ascent Trajectory')
plt.xlabel('Theta 0')
plt.ylabel('Theta 1')

plt.plot(all_thetas[:, 0], all_thetas[:, 1], 'r*-') 

plt.show()


def logistic(x):
    return 1 / (1 + np.exp(-x))

def grad_log_likelihood(theta, y, X):
    return X.T.dot(y - logistic(X.dot(theta)))

def hess_log_likelihood(theta, X):
    G = logistic(X.dot(theta)) * logistic(-X.dot(theta))
    return -(X * G).T.dot(X)

def newton_method(X, y, max_iter=10):
    m, n = X.shape
    theta = np.zeros((n, 1))
    thetas = np.zeros((n, max_iter + 1))
    thetas[:, 0:1] = theta
    
    for t in range(max_iter):
        grad = grad_log_likelihood(theta, y, X)
        try:
            hess = hess_log_likelihood(theta, X)
            step = np.linalg.solve(hess, -grad)
        except LinAlgError:
            step = np.linalg.pinv(hess).dot(-grad)
        theta += step
        thetas[:, t + 1:t + 2] = theta
    
    return theta, thetas

X_train_np = np.array(X_train)
y_train_reshaped = np.array(y_train).reshape(-1, 1)
final_theta_newt, all_thetas_newt = newton_method(X_train_np, y_train_reshaped, max_iter=10)

final_theta_newt, all_thetas_newt[:, :5]


plt.figure(figsize=(7, 7))
cp = plt.contourf(theta_0_grid, theta_1_grid, ll_grid, cmap='viridis')
plt.colorbar(cp)
plt.title('Log-likelihood Countour Plot with Newton\'s Method Trajectory')
plt.xlabel('Theta 0')
plt.ylabel('Theta 1')
plt.plot(all_thetas_newt[:, 0], all_thetas_newt[:, 1], 'b*-')

plt.show()