# Sprawozdanie z eksperymentów klasyfikacji binarnej
## Detekcja ataków DoS Hulk - porównanie algorytmów uczenia maszynowego

### 🎯 Cel eksperymentu
Porównanie skuteczności dwóch algorytmów uczenia maszynowego w detekcji ataków DoS Hulk:
- **Random Forest** (algorytm ensemble)
- **Logistic Regression** (algorytm liniowy)

### 📊 Charakterystyka datasetu
- **Próbki treningowe**: 468,859 (BENIGN: 307,811, DoS Hulk: 161,048)
- **Próbki testowe**: 200,948 (BENIGN: 131,872, DoS Hulk: 69,076)
- **Liczba cech**: 14 znormalizowanych cech sieciowych
- **Proporcje klas**: ~66% BENIGN, ~34% DoS Hulk

### 🔬 Metodologia

#### Techniki zastosowane:
1. **Preprocessing danych**:
   - StandardScaler - normalizacja cech do średniej=0, odchylenie=1
   - Usunięcie wartości nieskończonych i brakujących
   - Podział train/test (70%/30%)

2. **Optymalizacja hiperparametrów**:
   - **Grid Search** z 3-fold Cross Validation
   - **Random Forest**: testowano n_estimators (100,200), max_depth (10,20), min_samples_split (2,5), min_samples_leaf (1,2)
   - **Logistic Regression**: testowano C (0.1,1,10), penalty='l2', solver='lbfgs'

3. **System trackingu eksperymentów**:
   - Automatyczne zapisywanie wyników
   - Porównanie z poprzednimi eksperymentami
   - Tracking mejpszych parametrów

### 📈 Wyniki eksperymentów

#### Random Forest (exp_001, exp_002, exp_003)
```
Najlepsze parametry: {max_depth: 20, min_samples_leaf: 1, min_samples_split: 5, n_estimators: 200}

Metryki wydajności:
- Accuracy: 99.99%
- Precision: 99.99% 
- Recall: 99.99%
- F1-Score: 99.99%
- CV Score: 99.98% ± 0.01%
```

**Confusion Matrix**:
```
                Predicted
Actual    BENIGN   DoS Hulk
BENIGN    131,852      20     ← 20 fałszywych alarmów
DoS Hulk        7  69,069     ← 7 niewykrytych ataków
```

#### Logistic Regression (exp_004)
```
Najlepsze parametry: {C: 10, penalty: 'l2', solver: 'lbfgs'}

Metryki wydajności:
- Accuracy: 97.3%
- Precision: 97.3%
- Recall: 97.3%
- F1-Score: 97.3%
```

**Confusion Matrix**:
```
                Predicted
Actual    BENIGN   DoS Hulk
BENIGN    129,354    2,518   ← 2,518 fałszywych alarmów
DoS Hulk    2,899   66,177   ← 2,899 niewykrytych ataków
```

### 🔍 Analiza wyników

#### Porównanie algorytmów:

| Metryka | Random Forest | Logistic Regression | Różnica |
|---------|---------------|---------------------|---------|
| **Accuracy** | 99.99% | 97.3% | **+2.69%** |
| **False Positives** | 20 | 2,518 | **126x mniej** |
| **False Negatives** | 7 | 2,899 | **414x mniej** |
| **Praktyczne znaczenie** | Niemal idealna detekcja | Dobre, ale znacznie gorsze |

#### Analiza Feature Importance:

**Random Forest - najważniejsze cechy**:
1. **Bwd Packet Length Mean** (19.0%) - długość pakietów wstecznych
2. **Destination Port** (11.3%) - port docelowy
3. **Flow Packets/s** (10.1%) - pakiety na sekundę
4. **Flow Bytes/s** (9.4%) - bajty na sekundę
5. **Init_Win_bytes_forward** (9.3%) - okno TCP

**Logistic Regression - najważniejsze cechy**:
1. **Destination Port** (59.4) - dominująca cecha
2. **Total Backward Packets** (35.8)
3. **Flow Bytes/s** (28.8)
4. **Total Fwd Packets** (27.8)
5. **Idle Mean** (24.8)

### 💡 Kluczowe obserwacje

#### Przewagi Random Forest:
- **Niemal idealna klasyfikacja** - błąd <0.01%
- **Bardzo niski False Positive Rate** (0.015%)
- **Bardzo niski False Negative Rate** (0.01%)
- **Stabilność wyników** - identyczne metryki w 3 eksperymentach
- **Równoważne traktowanie cech** - najważniejsza cecha to tylko 19%

#### Ograniczenia Logistic Regression:
- **97x więcej fałszywych alarmów** niż Random Forest
- **414x więcej niewykrytych ataków**
- **Nadmierna zależność od Destination Port** (59% ważności)
- **Problemy z konwergencją** - wymagane zwiększenie max_iter

#### Interpretacja biznesowa:
- **Random Forest**: Gotowy do wdrożenia produkcyjnego - minimalne fałszywe alarmy
- **Logistic Regression**: Wymaga dodatkowej optymalizacji - za dużo błędów dla krytycznych systemów

### 🎯 Wnioski i rekomendacje

#### Zalecenie główne:
**Random Forest jest zdecydowanie lepszym wyborem** dla detekcji ataków DoS Hulk ze względu na:
- Niemal idealną skuteczność (99.99%)
- Minimalną liczbę fałszywych alarmów
- Stabilność wyników
- Odporność na overfitting

#### Dalsze kierunki badań:
1. **Testowanie na FTP Patator** - weryfikacja uniwersalności Random Forest
2. **Optymalizacja Logistic Regression** - inne feature engineering, regularization
3. **Porównanie z Autoencoderem** - analiza supervised vs unsupervised
4. **Testy na nowych typach ataków** - generalization capability

#### Znaczenie praktyczne:
- **Random Forest** może być wdrożony w systemach real-time
- **Koszt błędów**: 1 niewykryty atak na ~10k próbek vs 414 na 10k dla LR
- **Operacyjność**: Minimalne false positive = mniej niepotrzebnych interwencji

### 📋 Podsumowanie techniki

**System trackingu eksperymentów** umożliwił:
- Automatyczne porównywanie wyników
- Tracking najlepszych parametrów  
- Wizualizację ewolucji metryk
- Systematyczne dokumentowanie postępu

**Grid Search** z Cross Validation zapewnił:
- Obiektywną optymalizację hiperparametrów
- Unikanie overfittingu
- Statystyczną wiarygodność wyników

Eksperyment potwierdza przewagę algorytmów ensemble (Random Forest) nad metodami liniowymi (Logistic Regression) w zadaniach detekcji anomalii sieciowych.

## **🏆 PORÓWNANIE WYNIKÓW:**

| Aspekt | Random Forest (Binary) | Autoencoder | Logistic Regression |
|--------|------------------------|-------------|-------------------|
| **Approach** | Supervised | Unsupervised | Supervised |
| **Training data** | BENIGN + DoS Hulk | BENIGN only | BENIGN + DoS Hulk |
| **Accuracy** | **99.99%** | ~99.x% | 97.3% |
| **False Positives** | **20** | ~few hundred | 2,518 |
| **Complexity** | Medium | High | Low |
| **Interpretability** | High | Medium | High |



💡 Kiedy używać którego:
Klasyfikator binarny (Random Forest) - NAJLEPSZY gdy:

✅ Masz dane obu klas (normal + attack)
✅ Chcesz najwyższą skuteczność
✅ Znasz konkretny typ ataku
✅ Potrzebujesz interpretowalne wyniki

Autoencoder - LEPSZY gdy:

✅ Masz tylko dane normalne
✅ Chcesz wykrywać NIEZNANE typy ataków
✅ Anomalie mogą być bardzo różnorodne
✅ Nie masz labeled attack data

Logistic Regression - gdy:

✅ Potrzebujesz prosty, szybki model
✅ Interpretowalność > skuteczność
✅ Ograniczone zasoby obliczeniowe


















