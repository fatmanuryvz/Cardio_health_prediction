from fatmanurprojects import DataScience
import pandas as pd

# Veriyi oku
df = pd.read_csv(r"C:\Users\Fatmanur\Desktop\cardio_train.csv", sep=";")
df = DataScience.prepare_cardio_data(df)

# Eksik veri kontrolü ve dağılım görselleştirme
DataScience.show_missing_data(df)
DataScience.plot_distribution(df, "age_years")
DataScience.plot_distribution(df, "bmi")

# Özellikler ve model eğitimi
features = ["age_years", "bmi", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
model = DataScience.train_model(df, features)
DataScience.save_model(model)

# Modeli dosyadan yükle
model = DataScience.load_model()

# Kolesterol & Glikoz sınıflandırma fonksiyonları
def classify_cholesterol(value):
    if value < 200:
        return 1
    elif 200 <= value <= 239:
        return 2
    else:
        return 3

def classify_glucose(value):
    if value < 100:
        return 1
    elif 100 <= value <= 125:
        return 2
    else:
        return 3

# Kullanıcıdan veri al
def get_user_input():
    print("🩺 Lütfen aşağıdaki bilgileri giriniz:")

    age = int(input("Yaş (yıl): "))
    height = float(input("Boy (cm): "))
    weight = float(input("Kilo (kg): "))
    ap_hi = int(input("Büyük Tansiyon (ap_hi): "))
    ap_lo = int(input("Küçük Tansiyon (ap_lo): "))
    cholesterol_val = float(input("Kolesterol (mg/dL): "))
    gluc_val = float(input("Glikoz (mg/dL): "))
    smoke = int(input("Sigara kullanıyor musun? (0:Hayır, 1:Evet): "))
    alco = int(input("Alkol kullanıyor musun? (0:Hayır, 1:Evet): "))
    active = int(input("Fiziksel olarak aktif misin? (0:Hayır, 1:Evet): "))

    bmi = weight / ((height / 100) ** 2)

    data = pd.DataFrame([{
        "age_years": age,
        "bmi": bmi,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": classify_cholesterol(cholesterol_val),
        "gluc": classify_glucose(gluc_val),
        "smoke": smoke,
        "alco": alco,
        "active": active
    }])

    return data

# Tahmin yap
sample = get_user_input()
prediction = model.predict(sample)

print("\n🔎 Tahmin sonucu:", "🟥 KALP HASTALIĞI VAR" if prediction[0] == 1 else "🟩 KALP HASTALIĞI YOK")
