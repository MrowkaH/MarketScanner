# MarketScanner

Zbiór narzędzi ML do prognozowania kluczowych wskaźników makroekonomicznych. Każdy folder zawiera niezależny predictor, który analizuje historyczne dane z FRED API i przy pomocy uczenia maszynowego przewiduje kolejny odczyt danego wskaźnika.

## Zasada działania

Każdy predictor realizuje ten sam pipeline:

1. **Zbieranie danych** - pobiera 20+ lat danych z FRED (Federal Reserve Economic Data) API
2. **Inżynieria cech** - tworzy cechy ML: zmiany MoM/YoY, lagi, średnie kroczące, sezonowość
3. **Analiza** - identyfikuje korelacje między wskaźnikami a targetem
4. **Trening modeli** - trenuje modele regresji (punkt predykcji) i klasyfikacji (rozkład prawdopodobieństw)
5. **Predykcja** - generuje prognozę z uwzględnieniem rynkowego expected wyniku
6. **Raport** - zapisuje wyniki do PDF w folderze `results/`

## Predictors

| Folder | Wskaźnik | Opis |
|--------|----------|------|
| `CPI_Predictor/` | Consumer Price Index (CPI) | Prognoza zmian cen konsumenckich na podstawie korelacji z innymi wskaźnikami ekonomicznymi |
| `PPI_Predictor/` | Producer Price Index (PPI) | Prognoza zmian cen producenckich z użyciem regresji i klasyfikacji ML, zwraca rozkład prawdopodobieństw po binach co 0.1% |

## Quick Start

```bash
# 1. Aktywuj venv (lub użyj bezpośredniej ścieżki do pythona)
.\venv\Scripts\python.exe -m pip install fredapi pandas numpy scikit-learn python-dotenv matplotlib fpdf2

# 2. Ustaw FRED API key w pliku .env danego predictora
#    np. PPI_Predictor/.env → FRED_API_KEY=your_key

# 3. Uruchom wybrany predictor
.\venv\Scripts\python.exe -u PPI_Predictor\main.py --expected 0.3
```

## Struktura projektu

```
MarketScanner/
├── README.md               # Ten plik
├── requirements.txt        # Zależności Python
├── venv/                   # Wirtualne środowisko Python
├── CPI_Predictor/          # Predictor dla CPI
│   ├── main.py
│   ├── .env
│   ├── config/
│   ├── src/
│   └── data/
└── PPI_Predictor/          # Predictor dla PPI
    ├── main.py
    ├── .env
    ├── config/
    ├── src/
    ├── data/
    └── results/            # Raporty PDF
```

## FRED API Key

Klucz API można uzyskać za darmo na: https://fred.stlouisfed.org/docs/api/api_key.html

Każdy predictor ma własny plik `.env` z kluczem:
```
FRED_API_KEY=your_api_key_here
```