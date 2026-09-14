"""
ML(Machine Learning)
Supervised Learning (Denetimli Öğrenme = Featrue + Label)
Modelin hem giriş verileri (x)
hem de bu verilere ait doğru cevapları (Etiketleri[Label]: y)

Temel Yapı:
    X(Features/Özellikler) -> Model -> y (Label/Etiketler)

Örnek:
    - E-Posta spam mı değil mi?
    - Ev fiyatları ne kadar olur?
    - Bu müşteriye borç verilir mi?

Öğrenme türü olan LABEL vardır.
"""

"""
Bir öğrencinin:
1- Günlük çalışma saati
2- Derse katılım yüzdesi
bu bilgilere bakarak sınavı geçip geçemeyeceğini tahmin edelim.

Label:
    0 : Kaldı
    1 : Geçti
    
Kullanılan Algoritma:
    Logistic Regression
    
Kurulum:
    pip install numpy scikit-learn
    python -c "import numpy; import sklearn; print('Kurulum başarılı')"
    
    python -m pip install -r requirements.txt
    
    -m → module
    -c → command
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

def main():
    # X(Features/Özellikler)
    # Bir öğrencinin:
    # 1 = Günlük çalışma saati
    # 30 = Derse katılım yüzdesi
    X = np.array([
        [1, 30],
        [2, 40],
        [2, 50],
        [3, 55],
        [4, 60],
        [5, 65],
        [6, 75],
        [7, 85],
        [8, 90],
        [9, 95],
    ])
    # y (Label/Etiketler)
    # 0 = Kaldı, 1 = Geçti

    # NOT: Supervised Learning'in en önemli özelliği -> X verileriyle birlikte y etiketlerinin bulunmasıdır (SL)

    y = np.array([
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1
    ])

    print("--- SUPERVISED LEARNING (Features(+) Label(+))")
    print("\nX - Öğrenci Özellikleri(Features)")
    print(X)
    print("\ny - Label(Etiketler)")
    print(y)

    # Model Oluşturma
    """
    LogisticRegression bir sınıflandırma algoritmasıdır.
    İki tane sınıf vardı. (0 -> Kaldı, 1 -> Geçti)
    LogisticRegression, iki veya daha fazla sınıfın hangisine ait oluduğunu tahmin etmek için kullanılan sınıfın 
    algoritmasıdır.
    """

    model = LogisticRegression()

    # Model Eğitimi
    # Model hem özellikleri hem de doğru cevapları görsün
    # Bu ilişkide çalışma saati + katılım oranı -> Geçti/Kaldı

    model.fit(X, y)

    # Instance
    # Örnek: Öğrenci 6 saat çalışıyor, derse katılım %80

    new_student = np.array([[6, 80]])

    # Tahmin
    prediction = model.predict(new_student)[0]

    # Tahmin Olasılıkları
    probabilities = model.predict_proba(new_student)[0]

    print("\nYeni Öğrenci")
    print("Çalışma Saati: 6 saat")
    print("Derse Katılım: %80")

    print("\nModel Tahmini:", prediction)

    # Conditional
    if prediction == 1:
        print("Sonuç: Öğrencinin GEÇMESİ bekleniyor.")
    else:
        print("Sonuç: Öğrencinin KALMASI bekleniyor.")

    print("\nOlasılıklar:")
    print(f"Kalma Olasılığı: %{probabilities[0] * 100:.2f}")
    print(f"Geçme Olasılığı: %{probabilities[1] * 100:.2f}")

    # --- ÖZET ---
    print("\nÖZET")
    print("Supervised Learning LABEL vardır.")
    print("UNUTMA: Model, geçmişteki doğru cevapları öğrenir")
    print("Bu örnekte label: 0 = Kaldı, 1 = Geçti")


if __name__ == "__main__":
    main()
