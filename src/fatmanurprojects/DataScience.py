import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class DataScience:
    @staticmethod
    def mean(numbers):
        return sum(numbers) / len(numbers)

    @staticmethod
    def prepare_cardio_data(df):
        df["age_years"] = df["age"] // 365
        df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
        return df

    @staticmethod
    def show_missing_data(df):
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            print("✅ Hiçbir eksik veri yok.")
        else:
            print("⚠️ Eksik değerler:")
            print(missing)

    @staticmethod
    def plot_distribution(df, column):
        plt.figure(figsize=(8, 4))
        sns.histplot(data=df, x=column, kde=True)
        plt.title(f"Dağılım Grafiği: {column}")
        plt.xlabel(column)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def train_model(df, features, target="cardio"):
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print("✅ Doğruluk (Accuracy):", round(acc, 3))
        print("\n📊 Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
        return model

    @staticmethod
    def save_model(model, path="cardio_model.pkl"):
        joblib.dump(model, path)
        print(f"📁 Model kaydedildi: {path}")

    @staticmethod
    def load_model(path="cardio_model.pkl"):
        model = joblib.load(path)
        print(f"✅ Model yüklendi: {path}")
        return model
