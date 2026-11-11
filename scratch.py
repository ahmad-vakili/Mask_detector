# svm_mask_classifier.py
# پروژه: تشخیص ماسک صورت (Mask vs No Mask) با استفاده از SVM
# درس:هوش مصنوعی
# تهیه شده توسط: <احمد وکیلی>
#شماره دانشجویی <401103211>

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =========================
IMAGE_SIZE = 64  # تغییر اندازه همه تصاویر به 64×64
MAX_IMAGES = 500  # حداکثر تعداد تصویر در هر کلاس (برای سرعت آموزش)
RANDOM_STATE = 42  # برای تکرارپذیری نتایج
DATASET_DIR = 'dataset'  # پوشه‌ی حاوی with_mask و without_mask

# مسیر پوشه‌های هر کلاس
with_mask_dir = os.path.join(DATASET_DIR, 'with_mask')
without_mask_dir = os.path.join(DATASET_DIR, 'without_mask')


# =========================
def load_data(mask_dir, no_mask_dir, img_size=IMAGE_SIZE, max_count=MAX_IMAGES):
    X, y = [], []


    for i, fname in enumerate(os.listdir(mask_dir)[:max_count]):
        img_path = os.path.join(mask_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X.append(img.flatten())
        y.append(1)


    for i, fname in enumerate(os.listdir(no_mask_dir)[:max_count]):
        img_path = os.path.join(no_mask_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X.append(img.flatten())
        y.append(0)


    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y


# =========================
print("🔄 بارگذاری و پیش‌پردازش تصاویر ...")
X, y = load_data(with_mask_dir, without_mask_dir)
print(f"✅ تعداد کل تصاویر: {len(X)}")

# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y  # حفظ تناسب کلاس‌ها
)
print(f"✅ تقسیم داده‌ها: {len(X_train)} آموزش، {len(X_test)} تست")

# =========================
param_grid = {
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}
print("🔍 انجام GridSearchCV برای یافتن بهترین پارامترها ...")
grid = GridSearchCV(SVC(probability=True), param_grid, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
print("🏅 بهترین پارامترها:", grid.best_params_)

# بهترین مدل از GridSearch
best_model = grid.best_estimator_

# =========================
#آموزش مدل نهایی
# =========================
print("🧠 آموزش مدل نهایی با بهترین پارامترها ...")
best_model.fit(X_train, y_train)

# =========================
# پیش‌بینی بر روی داده‌های تست
# =========================
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

# =========================
#نمایش چند نمونه احتمال
# =========================
print("\n🔍 چند نمونه پیش‌بینی با احتمال:")
for i in range(5):
    print(f"نمونه {i + 1}: احتمال با ماسک = {y_proba[i][1]:.2f} | بدون ماسک = {y_proba[i][0]:.2f} → پیش‌بینی:",
          "با ماسک" if y_pred[i] == 1 else "بدون ماسک")
# =========================
#ارزیابی مدل
# =========================
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 دقت کلی: {acc:.2%}\n")
print("📊 گزارش طبقه‌بندی:")
print(classification_report(y_test, y_pred, target_names=['No Mask','Mask']))
print("🧩 ماتریس درهم‌ریختگی:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# نمایش ماتریس درهم‌ریختگی گرافیکی
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Mask','Mask'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# =========================
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
kern_acc = {}
for kernel in kernels:
    svm = SVC(kernel=kernel, C=1.0, gamma='scale')
    svm.fit(X_train, y_train)
    kern_acc[kernel] = accuracy_score(y_test, svm.predict(X_test))

plt.figure(figsize=(6,4))
plt.bar(kern_acc.keys(), kern_acc.values(), edgecolor='k')
plt.title('Accuracy by Kernel (default C=1, gamma=scale)')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# =========================
import joblib
model_path = 'svm_mask_model.pkl'
joblib.dump(best_model, model_path)
print(f"✅ مدل ذخیره شد در: {model_path}")