import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer, make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from matplotlib.colors import ListedColormap

class SVM:
    """Support Vector Machine with multiple kernel options"""
    
    def __init__(self, C=1.0, tol=1e-3, max_passes=10, kernel='linear', gamma=0.1, degree=3, coef0=1):
        """
        Initialize SVM
        
        Parameters:
        -----------
        C : float
            Regularization parameter
        tol : float
            Tolerance for stopping criterion
        max_passes : int
            Maximum number of passes over alphas without changing
        kernel : str
            Kernel type: 'linear', 'rbf', 'polynomial'
        gamma : float
            Kernel coefficient for 'rbf' and 'polynomial'
        degree : int
            Degree for polynomial kernel
        coef0 : float
            Independent term in polynomial kernel
        """
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.kernel_type = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        
    def kernel(self, x1, x2):
        """Compute kernel function"""
        if self.kernel_type == 'linear':
            return np.dot(x1, x2)
        elif self.kernel_type == 'rbf':
            # RBF (Gaussian) kernel: exp(-gamma * ||x1 - x2||^2)
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        elif self.kernel_type == 'polynomial':
            # Polynomial kernel: (gamma * <x1, x2> + coef0)^degree
            return (self.gamma * np.dot(x1, x2) + self.coef0) ** self.degree
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def fit(self, X, y):
        """
        Train SVM using SMO (Sequential Minimal Optimization) algorithm
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (must be -1 or 1)
        """
        self.X = X
        self.y = y.astype(float)
        m, n = X.shape
        
        self.alpha = np.zeros(m)
        self.b = 0
        
        # Cache for kernel computations
        self.kernel_cache = {}
        
        passes = 0
        iteration = 0
        max_iterations = m * self.max_passes
        
        print(f"Training SVM with {self.kernel_type} kernel...")
        
        while passes < self.max_passes and iteration < max_iterations:
            num_changed_alpha = 0
            
            for i in range(m):
                iteration += 1
                if iteration >= max_iterations:
                    break
                    
                Ei = self.predict_raw(self.X[i]) - self.y[i]
                
                # Check KKT conditions
                if ((self.y[i]*Ei < -self.tol and self.alpha[i] < self.C) or
                    (self.y[i]*Ei > self.tol and self.alpha[i] > 0)):
                    
                    # Select j randomly (not equal to i)
                    j = np.random.choice([x for x in range(m) if x != i])
                    Ej = self.predict_raw(self.X[j]) - self.y[j]
                    
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    
                    # Compute L and H (bounds for alpha[j])
                    if self.y[i] != self.y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[j] + self.alpha[i] - self.C)
                        H = min(self.C, self.alpha[j] + self.alpha[i])
                    
                    if L == H:
                        continue
                    
                    # Compute eta (second derivative of objective function)
                    eta = 2*self.kernel(X[i], X[j]) - self.kernel(X[i], X[i]) - self.kernel(X[j], X[j])
                    if eta >= 0:
                        continue
                    
                    # Update alpha_j
                    self.alpha[j] -= self.y[j] * (Ei - Ej) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Update alpha_i
                    self.alpha[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])
                    
                    # Compute b1 and b2
                    b1 = self.b - Ei \
                         - self.y[i]*(self.alpha[i] - alpha_i_old)*self.kernel(X[i], X[i]) \
                         - self.y[j]*(self.alpha[j] - alpha_j_old)*self.kernel(X[i], X[j])
                    
                    b2 = self.b - Ej \
                         - self.y[i]*(self.alpha[i] - alpha_i_old)*self.kernel(X[i], X[j]) \
                         - self.y[j]*(self.alpha[j] - alpha_j_old)*self.kernel(X[j], X[j])
                    
                    # Update b
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    num_changed_alpha += 1
            
            if num_changed_alpha == 0:
                passes += 1
            else:
                passes = 0
        
        # Store support vectors
        self.support_vector_indices = np.where(self.alpha > 1e-5)[0]
        self.support_vectors = self.X[self.support_vector_indices]
        self.support_vector_labels = self.y[self.support_vector_indices]
        self.support_vector_alphas = self.alpha[self.support_vector_indices]
        
        print(f"Training complete. Found {len(self.support_vector_indices)} support vectors.")
    
    def predict_raw(self, x):
        """Compute raw prediction (before sign)"""
        result = 0
        for i, (a, yi, Xi) in enumerate(zip(self.alpha, self.y, self.X)):
            if a > 0:
                result += a * yi * self.kernel(Xi, x)
        return result + self.b
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        predictions = []
        for x in X:
            predictions.append(np.sign(self.predict_raw(x)))
        return np.array(predictions)
    
    def decision_function(self, X):
        """Compute the decision function for samples in X"""
        return np.array([self.predict_raw(x) for x in X])


def plot_decision_boundary(X, y, model, title, ax=None):
    """Plot decision boundary and margins"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create mesh
    h = 0.02  # step size in mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    ax.contourf(xx, yy, Z, levels=[-np.inf, -1, 0, 1, np.inf], 
                colors=['#FFAAAA', '#AAAAFF', '#AAFFAA', '#AAAAFF'],
                alpha=0.3)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], 
               colors=['gray', 'black', 'gray'],
               linestyles=['--', '-', '--'],
               linewidths=[2, 3, 2])
    
    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', 
                        s=50, edgecolors='k', alpha=0.7)
    
    # Highlight support vectors
    if hasattr(model, 'support_vectors'):
        ax.scatter(model.support_vectors[:, 0], model.support_vectors[:, 1],
                  s=200, linewidth=2, facecolors='none', edgecolors='green',
                  label=f'Support Vectors ({len(model.support_vectors)})')
    
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_confusion_matrix(y_true, y_pred, title, ax=None):
    """Plot confusion matrix"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    return ax


def compare_kernels_on_dataset(X, y, dataset_name):
    """Compare different kernels on a dataset"""
    # Limit dataset size for faster training
    if len(X) > 150:
        indices = np.random.choice(len(X), 150, replace=False)
        X = X[indices]
        y = y[indices]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define kernels to test
    kernels = [
        ('Linear', {'kernel': 'linear', 'C': 1.0, 'max_passes': 5}),
        ('RBF', {'kernel': 'rbf', 'C': 1.0, 'gamma': 0.5, 'max_passes': 5}),
        ('Polynomial', {'kernel': 'polynomial', 'C': 1.0, 'gamma': 0.1, 'degree': 3, 'max_passes': 5})
    ]
    
    results = []
    
    # Create figure for visualization
    fig = plt.figure(figsize=(18, 12))
    
    for idx, (kernel_name, params) in enumerate(kernels):
        print(f"\n{'='*60}")
        print(f"Testing {kernel_name} Kernel on {dataset_name}")
        print(f"{'='*60}")
        
        # Train model
        model = SVM(**params)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        print(f"\nTraining Accuracy: {train_acc:.4f}")
        print(f"Testing Accuracy: {test_acc:.4f}")
        print(f"\nClassification Report (Test Set):")
        print(classification_report(y_test, y_test_pred))
        
        results.append({
            'Kernel': kernel_name,
            'Train Acc': train_acc,
            'Test Acc': test_acc,
            'Support Vectors': len(model.support_vector_indices)
        })
        
        # Visualizations (only for 2D data)
        if X.shape[1] == 2:
            # Decision boundary
            ax1 = plt.subplot(3, 3, idx*3 + 1)
            plot_decision_boundary(X_train_scaled, y_train, model, 
                                 f'{kernel_name} - Decision Boundary', ax1)
            
            # Confusion matrix - train
            ax2 = plt.subplot(3, 3, idx*3 + 2)
            plot_confusion_matrix(y_train, y_train_pred, 
                                f'{kernel_name} - Train CM', ax2)
            
            # Confusion matrix - test
            ax3 = plt.subplot(3, 3, idx*3 + 3)
            plot_confusion_matrix(y_test, y_test_pred, 
                                f'{kernel_name} - Test CM', ax3)
    
    plt.tight_layout()
    plt.savefig(f'svm_kernels_{dataset_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'svm_kernels_{dataset_name.lower().replace(' ', '_')}.png'")
    plt.show()
    
    # Print comparison table
    print(f"\n{'='*60}")
    print(f"KERNEL COMPARISON - {dataset_name}")
    print(f"{'='*60}")
    print(f"{'Kernel':<15} {'Train Acc':<12} {'Test Acc':<12} {'Support Vectors':<18}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['Kernel']:<15} {r['Train Acc']:<12.4f} {r['Test Acc']:<12.4f} {r['Support Vectors']:<18}")
    print(f"{'='*60}\n")


def visualize_kernel_effects():
    """Visualize how different kernels handle different data patterns"""
    print("\n" + "="*60)
    print("VISUALIZING KERNEL EFFECTS ON DIFFERENT DATA PATTERNS")
    print("="*60)
    
    # Create different datasets (smaller for faster training)
    datasets = [
        ('Linearly Separable', *make_classification(n_samples=100, n_features=2, 
                                                     n_redundant=0, n_informative=2,
                                                     n_clusters_per_class=1, 
                                                     class_sep=2.0, random_state=42)),
        ('Moons (Non-linear)', *make_moons(n_samples=100, noise=0.15, random_state=42)),
        ('Circles (Non-linear)', *make_circles(n_samples=100, noise=0.1, 
                                               factor=0.5, random_state=42))
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    
    kernels = [
        ('Linear', {'kernel': 'linear', 'C': 1.0, 'max_passes': 5}),
        ('RBF', {'kernel': 'rbf', 'C': 1.0, 'gamma': 1.0, 'max_passes': 5}),
        ('Polynomial', {'kernel': 'polynomial', 'C': 1.0, 'gamma': 0.5, 'degree': 3, 'max_passes': 5})
    ]
    
    for row, (dataset_name, X, y) in enumerate(datasets):
        # Convert labels to -1, 1
        y = np.where(y == 0, -1, 1)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for col, (kernel_name, params) in enumerate(kernels):
            ax = axes[row, col]
            
            # Train model
            model = SVM(**params)
            model.fit(X_scaled, y)
            
            # Get accuracy
            y_pred = model.predict(X_scaled)
            acc = accuracy_score(y, y_pred)
            
            # Plot
            plot_decision_boundary(X_scaled, y, model, 
                                 f'{dataset_name}\n{kernel_name} Kernel (Acc: {acc:.3f})', 
                                 ax)
    
    plt.tight_layout()
    plt.savefig('kernel_effects_comparison.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'kernel_effects_comparison.png'")
    plt.show()


def main():
    """Main function to run all experiments"""
    print("\n" + "="*70)
    print(" "*15 + "SVM WITH DIFFERENT KERNELS")
    print(" "*10 + "Educational Visualization and Comparison")
    print("="*70)
    
    # 1. Visualize kernel effects on different patterns
    visualize_kernel_effects()
    
    # 2. Test on Iris dataset (2 features, 2 classes)
    print("\n\n" + "="*70)
    print("EXPERIMENT 1: IRIS DATASET (Setosa vs Versicolor)")
    print("="*70)
    iris = load_iris()
    # Use only first 2 classes and first 2 features for visualization
    X_iris = iris.data[:100, :2]
    y_iris = iris.target[:100]
    y_iris = np.where(y_iris == 0, -1, 1)
    
    compare_kernels_on_dataset(X_iris, y_iris, "Iris Dataset")
    
    # 3. Test on synthetic non-linear data
    print("\n\n" + "="*70)
    print("EXPERIMENT 2: MOONS DATASET (Non-linear)")
    print("="*70)
    X_moons, y_moons = make_moons(n_samples=150, noise=0.15, random_state=42)
    y_moons = np.where(y_moons == 0, -1, 1)
    
    compare_kernels_on_dataset(X_moons, y_moons, "Moons Dataset")
    
    # 4. Test on circles dataset
    print("\n\n" + "="*70)
    print("EXPERIMENT 3: CIRCLES DATASET (Non-linear)")
    print("="*70)
    X_circles, y_circles = make_circles(n_samples=150, noise=0.1, 
                                        factor=0.5, random_state=42)
    y_circles = np.where(y_circles == 0, -1, 1)
    
    compare_kernels_on_dataset(X_circles, y_circles, "Circles Dataset")
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("Check the generated PNG files for visualizations.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
