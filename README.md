# Stock-analysis-with-AI

## Project Description
This project explores the application of Machine Learning, particularly Long Short-Term Memory (LSTM) networks, for forecasting stock prices. The developed solution aims to assist smaller investors in determining optimal holding periods for stocks by providing daily probability ratings for reaching a specified value.

## Authors
- Anchal Gera
- Leonard Katz
- Muralikrishna Naripeddi
- Leo Strauch

## Supervisors
- Prof. Dr. Thomas Burkhardt, University of Koblenz-Landau
- Dipl. Inf. Heiko Neuhaus, University of Koblenz-Landau

## Research Approach
The project follows the Design Science Research (DSR) methodology to develop a practical solution. After a comprehensive analysis of existing methods, an LSTM model was implemented to predict stock prices.

## Technical Details
### Data Source
- Historical stock price data (2019-2023)
- Financial data retrieved using the `yfinance` API

### Model Architecture
- 3 LSTM layers with dropout to reduce overfitting
- Implemented using TensorFlow/Keras
- Optimized using Adam optimizer
- Evaluated using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)

## Installation & Usage
### Requirements
- Python 3.x
- Dependencies:
  ```bash
  pip install tensorflow pandas numpy matplotlib scikit-learn yfinance
  ```

### Usage
1. Retrieve and prepare data
   ```python
   import yfinance as yf
   data = yf.Ticker("AAPL").history(period="5y")
   ```
2. Train and evaluate the model
   ```python
   model.fit(train_data, epochs=50, batch_size=32)
   mse, rmse = evaluate_model(model, test_data)
   ```
3. Visualize results
   ```python
   plt.plot(real_prices, label='Real Prices')
   plt.plot(predicted_prices, label='Predicted Prices')
   plt.legend()
   plt.show()
   ```

## Results
The trained LSTM model demonstrates solid forecasting performance with an RMSE of approximately 7.87 on the test dataset.

## License
This project is licensed under the MIT License.

## Contact
For any questions or suggestions, feel free to contact the authors.

## -------------------------------------German version---------------
# Aktienanalyse mit KI

## Projektbeschreibung
Dieses Projekt untersucht die Anwendung von Machine Learning, insbesondere Long Short-Term Memory (LSTM)-Netzwerken, zur Vorhersage von Aktienkursen. Die entwickelte Lösung soll insbesondere kleineren Investoren helfen, optimale Haltezeiten für Aktien zu bestimmen, indem sie tägliche Wahrscheinlichkeitsbewertungen für das Erreichen eines bestimmten Werts liefert.

## Autoren
- Anchal Gera
- Leonard Katz
- Muralikrishna Naripeddi
- Leo Strauch

## Betreuer
- Prof. Dr. Thomas Burkhardt, Universität Koblenz-Landau
- Dipl. Inf. Heiko Neuhaus, Universität Koblenz-Landau

## Forschungsansatz
Das Projekt folgt dem Design Science Research (DSR) Ansatz zur Entwicklung einer praktischen Lösung. Nach einer umfassenden Analyse der existierenden Methoden wurde ein LSTM-Modell implementiert, um Aktienkurse zu prognostizieren.

## Technische Details
### Datenquelle
- Historische Aktienkursdaten (2019-2023)
- Finanzdaten abgerufen mittels `yfinance` API

### Modellarchitektur
- 3 LSTM-Schichten mit Dropout zur Reduktion von Overfitting
- Implementiert mit TensorFlow/Keras
- Optimiert mittels Adam-Optimizer
- Bewertet mit Mean Squared Error (MSE) und Root Mean Squared Error (RMSE)

## Installation & Nutzung
### Voraussetzungen
- Python 3.x
- Abhängigkeiten:
  ```bash
  pip install tensorflow pandas numpy matplotlib scikit-learn yfinance
  ```

### Nutzung
1. Daten abrufen und vorbereiten
   ```python
   import yfinance as yf
   data = yf.Ticker("AAPL").history(period="5y")
   ```
2. Modell trainieren und bewerten
   ```python
   model.fit(train_data, epochs=50, batch_size=32)
   mse, rmse = evaluate_model(model, test_data)
   ```
3. Ergebnisse visualisieren
   ```python
   plt.plot(real_prices, label='Real Prices')
   plt.plot(predicted_prices, label='Predicted Prices')
   plt.legend()
   plt.show()
   ```

## Ergebnisse
Das trainierte LSTM-Modell zeigt eine solide Prognoseleistung mit einer RMSE von ca. 7.87 auf dem Testdatensatz.

## Lizenz
Dieses Projekt ist unter der MIT-Lizenz lizenziert.

## Kontakt
Bei Fragen oder Anregungen wenden Sie sich gerne an die Autoren.

