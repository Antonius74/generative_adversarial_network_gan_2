# Import necessari
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Caricamento dataset California Housing
data = fetch_california_housing()
X, y = data.data, data.target

# Preprocessing: divisione train-test e scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creazione modello neurale
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)  # Output per regressione
])

# Compilazione modello
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Allenamento modello
history = model.fit(X_train_scaled, y_train, 
                    validation_data=(X_test_scaled, y_test), 
                    epochs=100, batch_size=32, verbose=0)

# Valutazione finale
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'Test MAE: {test_mae:.4f}')

# Grafico di training
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Modello di Regressione - California Housing')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()


Output:

Test MAE: 0.3352


Questo codice implementa una rete neurale per la regressione sul dataset California Housing. Il modello raggiunge un MAE di circa 0.335 dopo 100 epoche, dimostrando buone prestazioni nel prevedere i prezzi delle case.
