# SVM with Multiple Kernels - Educational Implementation

## Quick Start

```bash
python svms_kernals.py
```

## What This Code Does

This implementation demonstrates Support Vector Machines with three different kernel types:

- **Linear Kernel**: For linearly separable data
- **RBF (Radial Basis Function) Kernel**: For non-linear patterns
- **Polynomial Kernel**: For polynomial decision boundaries

## Features

✅ **Multiple Kernels**: Compare Linear, RBF, and Polynomial kernels  
✅ **Real Datasets**: Iris, Moons, and Circles datasets  
✅ **Visualizations**: Decision boundaries, confusion matrices, support vectors  
✅ **Performance Metrics**: Accuracy, precision, recall, F1-score  
✅ **Educational**: Clear code structure with comments

## Generated Visualizations

After running, you'll get 4 PNG files:

1. `kernel_effects_comparison.png` - Overview of all kernels on different patterns
2. `svm_kernels_iris_dataset.png` - Results on Iris dataset
3. `svm_kernels_moons_dataset.png` - Results on Moons dataset
4. `svm_kernels_circles_dataset.png` - Results on Circles dataset

## Key Concepts

### Support Vectors

Points closest to the decision boundary (shown as green circles in plots)

### Decision Boundary

The line/curve that separates the two classes (black line)

### Margins

The distance from the decision boundary to the nearest points (gray dashed lines)

### When to Use Each Kernel

| Kernel | Best For | Speed | Complexity |
|--------|----------|-------|------------|
| Linear | Linearly separable data, text | Fast | Low |
| RBF | Non-linear data, general use | Medium | Medium |
| Polynomial | Polynomial relationships | Slow | High |

## Customization

You can modify parameters in the code:

- `C`: Regularization parameter (higher = less regularization)
- `gamma`: Kernel coefficient for RBF/Polynomial
- `degree`: Degree for polynomial kernel
- `max_passes`: Training iterations

## Dependencies

```bash
pip install numpy matplotlib scikit-learn seaborn
```

## Example Output

```
Training SVM with linear kernel...
Training complete. Found 8 support vectors.
Training Accuracy: 0.9857
Testing Accuracy: 1.0000
```
