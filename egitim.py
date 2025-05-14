import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Veriyi oku
df = pd.read_csv("veriseti.csv")

# Sütun isimlerini yeniden sırala: f0, f1, ..., f955, Etiket
df.columns = [f"f{i}" for i in range(956)] + ["Etiket"]

# Özellikler ve etiketleri ayır
X = df.drop("Etiket", axis=1)
y = df["Etiket"]

# Eğitim ve test verisine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluştur
model = Pipeline([
    ("ölçekleme", StandardScaler()),
    ("sınıflandırıcı", LogisticRegression(max_iter=1000))
])

# Eğit ve değerlendir
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
dogruluk = accuracy_score(y_test, y_pred)
print(f"Model doğruluk oranı: {dogruluk:.2f}")

# Modeli kaydet
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
