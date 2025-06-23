# PLVMUG25_showcase
Repozytorium z kodem zaprezentowanym w trakcie mojego wystąpienia na PLVMUG 25Q2

### Struktura
- `src/net_traffic_forecast.py` - prosta implementacja sieci LSTM do przewidywania obciążenia na serwerze
    - `data/daily-website-visitors.csv` - syntetyczny zbiór danych wykorzystany do trenowania sieci LSTM
- `src/LLM_showcase.ipynb` - notatnik Jupyter pokazujący, jak łatwo można dostrajać modele językowe z wykorzystaniem narzędzia `Trainer` z biblioteki HuggingFace

### Instalacja
Po sklonowaniu repozytorium, wystarczy w katalogu roboczym wywołać polecenie
```bash
pip install -r requirements.txt
```

To zainstaluje wszystkie potrzebne do uruchomienia programu i przejścia przez notatnik zależności (uwaga, będzie to dość długi proces m. in. ze względu na rozmiar biblioteki PyTorch)

