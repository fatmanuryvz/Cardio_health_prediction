from fatmanurprojects import DataScience
import pandas as pd

df = pd.read_csv(r"C:\Users\Fatmanur\Desktop\cardio_train.csv", sep=";")
df = DataScience.prepare_cardio_data(df)

# Eksik veri kontrolü
DataScience.show_missing_data(df)

# Dağılım görselleştirme
DataScience.plot_distribution(df, "age_years")
DataScience.plot_distribution(df, "bmi")
features = ["age_years", "bmi", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
model = DataScience.train_model(df, features)

# Modeli eğit
features = ["age_years", "bmi", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
model = DataScience.train_model(df, features)

# Eğitilen modeli kaydet
DataScience.save_model(model)
# Kaydedilen modeli geri yükle
model = DataScience.load_model()

# Modeli dosyadan yükle
model = DataScience.load_model()

# Örnek test verisi (tek kişi bilgisi)
sample = pd.DataFrame([{
    "age_years": 55,
    "bmi": 27.5,
    "ap_hi": 130,
    "ap_lo": 85,
    "cholesterol": 2,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1
}])

# Tahmin yap
prediction = model.predict(sample)
print("🩺 Kalp Hastalığı Tahmini:", "VAR" if prediction[0] == 1 else "YOK")


# Modeli yükle ------------------------------------
model = DataScience.load_model()

# Kullanıcıdan giriş al
def get_user_input():
    print("🩺 Lütfen aşağıdaki bilgileri giriniz:")

    age = int(input("Yaş (yıl): "))
    height = float(input("Boy (cm): "))
    weight = float(input("Kilo (kg): "))
    ap_hi = int(input("Büyük Tansiyon (ap_hi): "))
    ap_lo = int(input("Küçük Tansiyon (ap_lo): "))
    cholesterol = int(input("Kolesterol (1:Normal, 2:Orta, 3:Yüksek): "))
    gluc = int(input("Glikoz (1:Normal, 2:Orta, 3:Yüksek): "))
    smoke = int(input("Sigara kullanıyor musun? (0:Hayır, 1:Evet): "))
    alco = int(input("Alkol kullanıyor musun? (0:Hayır, 1:Evet): "))
    active = int(input("Fiziksel olarak aktif misin? (0:Hayır, 1:Evet): "))

    # BMI hesapla
    bmi = weight / ((height / 100) ** 2)

    # Tek satırlık DataFrame oluştur
    data = pd.DataFrame([{
        "age_years": age,
        "bmi": bmi,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active
    }])
    
    return data

# Veri al, tahmin yap
sample = get_user_input()
prediction = model.predict(sample)

# Sonucu yazdır
print("\n🔎 Tahmin sonucu:", "🟥 KALP HASTALIĞI VAR" if prediction[0] == 1 else "🟩 KALP HASTALIĞI YOK")
