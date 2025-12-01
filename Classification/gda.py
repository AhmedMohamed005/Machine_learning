import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as GDA
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
# 1. تحميل البيانات
iris = load_iris()
# نستخدم ميزتين فقط: طول البتلة (Petal Length) وعرض البتلة (Petal Width)
X = iris.data[:, [2, 3]]
y = iris.target

# البيانات الأصلية تحتوي على 3 فئات: 0, 1, 2
# GDA يمكنه التعامل مع التصنيف متعدد الفئات (Multiclass Classification)
print(f"Predicted classes: {iris.target_names}")
print(f"Shape of the dataset (samples, features): {X.shape}")

# 2. تقسيم البيانات (تدريب واختبار)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# 3. بناء نموذج GDA
# نستخدم LDA لأنه الحالة الشائعة والمناسبة لمجموعات البيانات التي تتبع التوزيع الغاوسي
classifier_gda = GDA()
classifier_qda = QDA()

# 4. تدريب النموذج
# يقوم النموذج بتقدير 3 مجموعات من التوزيعات الغاوسية (واحدة لكل فئة)
classifier_gda.fit(X_train, y_train)
classifier_qda.fit(X_train, y_train)

# 5. التنبؤ بالنتائج
y_pred = classifier_gda.predict(X_test)
y_pred_qda = classifier_qda.predict(X_test)

# 6. تقييم الأداء
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
accuracy_qda = accuracy_score(y_test, y_pred_qda)
cm_qda = confusion_matrix(y_test, y_pred_qda)

print("\n--- Performance Evaluation ---")
print(f"Accuracy: {accuracy*100:.2f}%")
print("Confusion Matrix:")
print(cm)
print(f"\nConfusion Matrix Interpretation:\n Rows are the true values, Columns are the predictions (0:setosa, 1:versicolor, 2:virginica)")

print("\n--- Performance Evaluation ---")
print(f"Accuracy: {accuracy_qda*100:.2f}%")
print("Confusion Matrix:")
print(cm_qda)
print(f"\nConfusion Matrix Interpretation:\n Rows are the true values, Columns are the predictions (0:setosa, 1:versicolor, 2:virginica)")

# 7. التمثيل البصري للحدود الفاصلة
def visualize_gda_multiclass(X, y, classifier, title):
    X_set, y_set = X, y
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    
    # التنبؤ بنقاط الشبكة (لتلوين الخلفية)
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
    
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    
    # رسم نقاط البيانات الفعلية
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green', 'blue'))(i), label = iris.target_names[j])
    
    plt.title(title)
    plt.xlabel('Petal Length ')
    plt.ylabel('Petal Width ')
    plt.legend()
    plt.show()

def visualize_qda_multiclass(X, y, classifier, title):
    X_set, y_set = X, y
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    
    # التنبؤ بنقاط الشبكة (لتلوين الخلفية)
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
    
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    
    # رسم نقاط البيانات الفعلية
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green', 'blue'))(i), label = iris.target_names[j])
    
    plt.title(title)
    plt.xlabel('Petal Length ')
    plt.ylabel('Petal Width ')
    plt.legend()
    plt.show()

visualize_gda_multiclass(X_train, y_train, classifier_gda, 'Gaussian Discriminant Analysis (LDA) on Iris Data')
visualize_qda_multiclass(X_train, y_train, classifier_qda, 'Quadratic Discriminant Analysis (QDA) on Iris Data')
