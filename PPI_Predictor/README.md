# PPI Predictor

Machine learning tool that predicts the monthly Producer Price Index (PPI) change by analyzing 20+ years of historical economic data from the FRED API.

## What it does

- Pobiera dane z ostatnich 20 lat dotyczące PPI oraz 18 powiązanych wskaźników ekonomicznych (ropa WTI, CPI, zatrudnienie, produkcja przemysłowa, import/eksport, podaż pieniądza M2 i inne)
- Tworzy ~140 cech inżynieryjnych: zmiany MoM%, lagi (1-12 mies.), średnie kroczące, odchylenia standardowe, sezonowość cykliczna
- Trenuje modele **regresji** (Gradient Boosting, Random Forest, Ridge) dające punkt predykcji
- Trenuje modele **klasyfikacji** (GB Classifier, RF Classifier) generujące rozkład prawdopodobieństw po binach co 0.1%
- Uwzględnia aktualny rynkowy **expected** wynik jako dodatkową cechę
- Generuje raport PDF z tabelą prawdopodobieństw, wykresem dystrybucji i metrykami modeli

## Requirements

```
fredapi, pandas, numpy, scikit-learn, python-dotenv, matplotlib, fpdf2
```

Instalacja (z głównego folderu MarketScanner):

```bash
.\venv\Scripts\python.exe -m pip install fredapi pandas numpy scikit-learn python-dotenv matplotlib fpdf2
```

## Configuration

Plik `.env` w folderze `PPI_Predictor/` musi zawierać klucz API FRED:

```
FRED_API_KEY=your_api_key_here
```

Klucz FRED API można uzyskać za darmo na: https://fred.stlouisfed.org/docs/api/api_key.html

Parametry modelu (biny, zakres historii, features) można dostosować w `config/settings.py`.

## Usage

Z folderu `MarketScanner/`:

```bash
# Z podanym expected (np. rynek oczekuje +0.3% MoM)
.\venv\Scripts\python.exe -u PPI_Predictor\main.py --expected 0.3

# Bez expected - program zapyta interaktywnie
.\venv\Scripts\python.exe -u PPI_Predictor\main.py
```

## Output

Program wyświetla w konsoli:
- Analizę korelacji i wzorce sezonowe
- Metryki modeli (MAE, RMSE, R², Accuracy, Log Loss)
- Punkt predykcji regresji (ensemble) z przedziałem ufności
- Tabelę rozkładu prawdopodobieństw (każdy bin co 0.1% z szansą %)

Dodatkowo generuje **raport PDF** w folderze `results/` zawierający:
- Podsumowanie historyczne
- Wyniki regresji
- Kolorową tabelę prawdopodobieństw
- Wykres słupkowy dystrybucji

## Project Structure

```
PPI_Predictor/
├── main.py                  # Entry point - orchestracja pipeline
├── .env                     # FRED API key
├── config/
│   └── settings.py          # Parametry: serie FRED, lagi, modele, biny
├── src/
│   ├── data_collector.py    # Pobieranie danych z FRED API
│   ├── data_processor.py    # Inżynieria cech (lagi, rolling, sezonowość)
│   ├── analyzer.py          # Analiza korelacji i wzorców historycznych
│   ├── predictor.py         # Modele ML (regresja + klasyfikacja)
│   └── report_generator.py  # Generator raportów PDF
├── data/
│   ├── raw.py               # Placeholder dla surowych danych
│   └── processed.py         # Placeholder dla przetworzonych danych
└── results/                 # Wygenerowane raporty PDF
```
