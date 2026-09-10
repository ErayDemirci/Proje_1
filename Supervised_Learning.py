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
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

def main():
    # X(Features/Özellikler)
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




