import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# --- A. Data Loading and Preparation ---
# 1. Load the data
iris = load_iris()
# Use only two features for 2D visualization (Petal Length and Petal Width)
X = iris.data[:, [2, 3]]
print(f"Shape of the dataset (samples, features): {X.shape}")
print(f"Predicted classes: {iris.target_names}")
y = iris.target
print(f"Shape of the target array: {y.shape}")
print(f"Target names: {iris.target}")
target_names = iris.target_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# 2. Feature Scaling (Crucial for SVM!)
# SVM relies on distances, so features must be on the same scale.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --- B. Model Training (The Kernel Trick) ---
# 3. Create the SVM Model
# kernel='rbf' activates the Kernel Trick, allowing for non-linear separation.
# C controls the penalty for misclassification (a hyperparameter).
# gamma defines how far the influence of a single training example reaches.
classifier_svm = SVC(kernel='rbf', C=10, gamma=0.1, random_state=42)

# 4. Train the Model
classifier_svm.fit(X_train, y_train)    

# 5. Make Predictions
y_pred = classifier_svm.predict(X_test)

# 6. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM (RBF Kernel) Classification Accuracy: {accuracy*100:.2f}%")
print(f"Number of Support Vectors: {len(classifier_svm.support_vectors_)}")


# --- C. Visualization of Decision Boundary ---
# Function to plot the decision boundary
def visualize_svm_boundary(X, y, classifier, title):
    X_set, y_set = X, y
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    
    # Predict over the grid (to color the background)
    Z = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
    plt.contourf(X1, X2, Z, alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
    
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    
    # Plot the actual data points
    colors = ('red', 'green', 'blue')
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = colors[i], label = target_names[j], edgecolors='k')

    # Highlight the Support Vectors
    # Support Vectors are the points that define the margin
    sv = classifier.support_vectors_
    plt.scatter(sv[:, 0], sv[:, 1], s=180, facecolors='none', edgecolors='k', linewidth=1.5, label='Support Vectors')

    plt.title(title)
    plt.xlabel('Petal Length (Scaled)')
    plt.ylabel('Petal Width (Scaled)')
    plt.legend()
    plt.show()

# Visualize the Training Set results
visualize_svm_boundary(X_train, y_train, classifier_svm, 'SVM (RBF Kernel) on Iris Data')