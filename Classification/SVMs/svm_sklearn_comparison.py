"""
SVM Kernel Comparison using Scikit-Learn
Fast implementation comparing Linear, RBF, and Polynomial kernels
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_moons, make_circles, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from matplotlib.colors import ListedColormap
import time


def plot_decision_boundary(X, y, model, title, ax):
    """Plot decision boundary for 2D data"""
    h = 0.02  # mesh step size
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.contour(xx, yy, Z, colors='black', linewidths=1, levels=[0.5])
    
    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', 
                        s=50, edgecolors='k', alpha=0.8)
    
    # Highlight support vectors
    if hasattr(model, 'support_vectors_'):
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                  s=200, linewidth=2, facecolors='none', edgecolors='lime',
                  label=f'Support Vectors ({len(model.support_vectors_)})')
    
    ax.set_xlabel('Feature 1', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature 2', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)


def plot_confusion_matrix_heatmap(y_true, y_pred, title, ax):
    """Plot confusion matrix as heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                cbar_kws={'label': 'Count'}, square=True)
    ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')


def compare_svm_kernels(X, y, dataset_name):
    """Compare different SVM kernels on a dataset"""
    print(f"\n{'='*70}")
    print(f"  {dataset_name.upper()}")
    print(f"{'='*70}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features (important for SVM!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define kernels to test
    kernels = {
        'Linear': SVC(kernel='linear', C=1.0, random_state=42),
        'RBF': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
        'Polynomial': SVC(kernel='poly', degree=3, C=1.0, gamma='scale', random_state=42)
    }
    
    results = []
    
    # Create figure for 2D visualization
    if X.shape[1] == 2:
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        fig.suptitle(f'SVM Kernel Comparison - {dataset_name}', 
                     fontsize=16, fontweight='bold', y=0.995)
    
    for idx, (kernel_name, model) in enumerate(kernels.items()):
        print(f"\n{'-'*70}")
        print(f"  {kernel_name} Kernel")
        print(f"{'-'*70}")
        
        # Train
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time
        
        # Predict
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        n_support = len(model.support_vectors_)
        
        print(f"Training Time:     {train_time:.4f} seconds")
        print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"Testing Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"Support Vectors:   {n_support}")
        
        print(f"\nClassification Report (Test Set):")
        print(classification_report(y_test, y_test_pred, zero_division=0))
        
        results.append({
            'Kernel': kernel_name,
            'Train Acc': train_acc,
            'Test Acc': test_acc,
            'Support Vectors': n_support,
            'Time (s)': train_time
        })
        
        # Visualizations for 2D data
        if X.shape[1] == 2:
            # Decision boundary
            ax1 = axes[idx, 0]
            plot_decision_boundary(X_train_scaled, y_train, model, 
                                 f'{kernel_name} - Training Data', ax1)
            
            # Confusion matrix - train
            ax2 = axes[idx, 1]
            plot_confusion_matrix_heatmap(y_train, y_train_pred, 
                                         f'{kernel_name} - Train CM', ax2)
            
            # Confusion matrix - test
            ax3 = axes[idx, 2]
            plot_confusion_matrix_heatmap(y_test, y_test_pred, 
                                         f'{kernel_name} - Test CM', ax3)
    
    # Save and show plot
    if X.shape[1] == 2:
        plt.tight_layout()
        filename = f'sklearn_svm_{dataset_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n✅ Visualization saved as '{filename}'")
        plt.show()
    
    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  SUMMARY - {dataset_name}")
    print(f"{'='*70}")
    print(f"{'Kernel':<12} {'Train Acc':<12} {'Test Acc':<12} {'SV':<8} {'Time (s)':<10}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['Kernel']:<12} {r['Train Acc']:<12.4f} {r['Test Acc']:<12.4f} "
              f"{r['Support Vectors']:<8} {r['Time (s)']:<10.4f}")
    print(f"{'='*70}\n")
    
    return results


def create_kernel_comparison_grid():
    """Create a comprehensive 3x3 grid comparing kernels on different datasets"""
    print("\n" + "="*70)
    print("  SVM KERNEL COMPARISON - COMPREHENSIVE VISUALIZATION")
    print("="*70)
    
    # Create datasets (small for fast execution)
    datasets = [
        ('Linearly Separable', *make_classification(n_samples=150, n_features=2, 
                                                     n_redundant=0, n_informative=2,
                                                     n_clusters_per_class=1, 
                                                     class_sep=2.0, random_state=42)),
        ('Moons', *make_moons(n_samples=150, noise=0.15, random_state=42)),
        ('Circles', *make_circles(n_samples=150, noise=0.1, factor=0.5, random_state=42))
    ]
    
    kernels = ['linear', 'rbf', 'poly']
    kernel_names = ['Linear', 'RBF', 'Polynomial']
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle('SVM Kernels: Complete Comparison', fontsize=18, fontweight='bold', y=0.995)
    
    for row, (dataset_name, X, y) in enumerate(datasets):
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for col, (kernel, kernel_name) in enumerate(zip(kernels, kernel_names)):
            ax = axes[row, col]
            
            # Train model
            if kernel == 'poly':
                model = SVC(kernel=kernel, degree=3, C=1.0, gamma='scale', random_state=42)
            else:
                model = SVC(kernel=kernel, C=1.0, gamma='scale', random_state=42)
            
            model.fit(X_scaled, y)
            
            # Get accuracy
            y_pred = model.predict(X_scaled)
            acc = accuracy_score(y, y_pred)
            
            # Plot
            title = f'{dataset_name}\n{kernel_name} (Acc: {acc:.3f})'
            plot_decision_boundary(X_scaled, y, model, title, ax)
    
    plt.tight_layout()
    plt.savefig('sklearn_svm_complete_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✅ Complete comparison saved as 'sklearn_svm_complete_comparison.png'")
    plt.show()


def main():
    """Main function to run all experiments"""
    print("\n" + "="*70)
    print("  SCIKIT-LEARN SVM KERNEL COMPARISON")
    print("  Fast Implementation with Professional Visualizations")
    print("="*70)
    
    # 1. Create comprehensive grid first
    create_kernel_comparison_grid()
    
    # 2. Iris Dataset (2 features, 2 classes for visualization)
    print("\n\n" + "="*70)
    print("  EXPERIMENT 1: IRIS DATASET")
    print("="*70)
    iris = load_iris()
    # Use only first 2 classes and first 2 features
    X_iris = iris.data[:100, :2]
    y_iris = iris.target[:100]
    
    compare_svm_kernels(X_iris, y_iris, "Iris Dataset (Setosa vs Versicolor)")
    
    # 3. Moons Dataset
    print("\n\n" + "="*70)
    print("  EXPERIMENT 2: MOONS DATASET (Non-linear)")
    print("="*70)
    X_moons, y_moons = make_moons(n_samples=200, noise=0.15, random_state=42)
    
    compare_svm_kernels(X_moons, y_moons, "Moons Dataset")
    
    # 4. Circles Dataset
    print("\n\n" + "="*70)
    print("  EXPERIMENT 3: CIRCLES DATASET (Highly Non-linear)")
    print("="*70)
    X_circles, y_circles = make_circles(n_samples=200, noise=0.1, 
                                        factor=0.5, random_state=42)
    
    compare_svm_kernels(X_circles, y_circles, "Circles Dataset")
    
    print("\n" + "="*70)
    print("  ✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("  📊 Check the generated PNG files for visualizations")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
