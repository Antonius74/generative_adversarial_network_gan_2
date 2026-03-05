# Generative Adversarial Network (GAN): Una Guida Didattica

## 1. Introduzione e Teoria

### Cos'è una GAN?
Una Generative Adarial Network (GAN) è un'architettura di machine learning composta da due reti neurali che competono tra loro: un generatore e un discriminatore. L'obiettivo del generatore è creare dati falsi che sembrino autentici, mentre il discriminatore cerca di distinguere i dati reali da quelli falsi.

### Come funziona l'algoritmo?
Immagina un'artista (generatore) che imita dipinti famosi e un critico d'arte (discriminatore) che deve riconoscere i falsi. Inizialmente, l'artista fa errori grossolani e il critico li smaschera facilmente. Con il tempo, l'artista migliora osservando i dipinti reali e ascoltando le critiche, mentre anche il critico diventa più abile nel riconoscere le imitazioni.

### Esempio pratico semplice
Supponiamo di voler generare numeri "7" che assomiglino a quelli scritti a mano. Il generatore crea forme casuali che gradualmente diventano numeri "7" sempre più realistici. Il discriminatore riceve sia numeri "7" veri dal dataset MNIST che quelli falsi del generatore, e deve indovinare quali sono autentici.

### Formulazione matematica
La GAN può essere vista come un gioco a somma zero tra due giocatori. La funzione obiettivo è:

$$
\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1-D(G(z)))]
$$

Dove:
- $G$: funzione del generatore
- $D$: funzione del discriminatore
- $p_{data}$: distribuzione dei dati reali
- $p_z$: distribuzione del rumore di input
- $x$: campione reale
- $z$: vettore di rumore

## 2. Dati utilizzati (Input/Output)

### Dataset MNIST
Il dataset MNIST contiene 70.000 immagini di cifre scritte a mano, ciascuna di dimensioni 28×28 pixel in scala di grigi.

### Caratteristiche degli input
- **Dati reali**: Immagini 28×28 pixel normalizzate tra -1 e 1
- **Rumore per il generatore**: Vettori casuali di dimensione 100 (distribuzione normale)
- **Dimensioni**: 
  - Batch size: tipicamente 64-128
  - Canali: 1 (scala di grigi)
  - Risoluzione: 28×28 pixel

### Esempio numerico di dato grezzo
Un pixel nel dataset MNIST originale ha valori tra 0 e 255. Dopo la normalizzazione:

$$
pixel_{normalizzato} = \frac{pixel_{originale} - 127.5}{127.5}
$$

Esempio: se un pixel ha valore 200:
$$
\frac{200 - 127.5}{127.5} \approx 0.57
$$

### Output atteso
- **Generatore**: Produce immagini fake che imitano quelle del MNIST
- **Discriminatore**: Fornisce probabilità (0-1) che un'input sia reale

## 3. Analisi del codice

### Struttura generale del codice GAN

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
```

### Preprocessing dei dati

```python
# Trasformazioni applicate alle immagini
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Caricamento del dataset MNIST
train_dataset = datasets.MNIST(root='./data', train=True, 
                              download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                         batch_size=64, shuffle=True)
```

Le trasformazioni convertono le immagini in tensori e le normalizzano nel range [-1, 1]. Questo è fondamentale perché le funzioni di attivazione tanh (usata nel generatore) produce output in questo range.

### Architettura del Generatore

```python
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 28*28),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.main(z).view(-1, 1, 28, 28)
```

Il generatore prende un vettore di rumore z (dimensione 100) e lo trasforma gradualmente in un'immagine 28×28. La sequenza è:
1. Proiezione lineare da 100 a 128 dimensioni
2. Attivazione LeakyReLU (evita neuroni morti)
3. Batch normalization (stabilizza l'addestramento)
4. Espansione a 512 dimensioni
5. Riduzione a 784 dimensioni (28×28)
6. Attivazione tanh per output in [-1, 1]

### Architettura del Discriminatore

```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.main(x)
```

Il discriminatore fa il percorso inverso:
1. Appiattisce l'immagine 28×28 in 784 valori
2. Riduce la dimensionalità a 512, poi 256
3. Output singolo con sigmoid per probabilità [0, 1]

### Training Loop

```python
def train_gan(generator, discriminator, dataloader, num_epochs):
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    
    for epoch in range(num_epochs):
        for batch_idx, (real_imgs, _) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            
            # Addestramento Discriminatore
            d_optimizer.zero_grad()
            
            # Dati reali
            real_labels = torch.ones(batch_size, 1)
            real_output = discriminator(real_imgs)
            d_loss_real = criterion(real_output, real_labels)
            
            # Dati falsi
            z = torch.randn(batch_size, latent_dim)
            fake_imgs = generator(z)
            fake_labels = torch.zeros(batch_size, 1)
            fake_output = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Addestramento Generatore
            g_optimizer.zero_grad()
            z = torch.randn(batch_size, latent_dim)
            fake_imgs = generator(z)
            output = discriminator(fake_imgs)
            g_loss = criterion(output, real_labels)  # Ingannare il D
            
            g_loss.backward()
            g_optimizer.step()
```

### Spiegazione del processo di training

**Fase 1: Addestramento Discriminatore**
1. Reset gradiente: `zero_grad()`
2. Calcola loss su dati reali: il D dovrebbe outputtare 1
3. Genera dati falsi: rumore → generatore → immagini fake
4. Calcola loss su dati falsi: il D dovrebbe outputtare 0
5. Backpropagazione e aggiornamento pesi

**Fase 2: Addestramento Generatore**
1. Reset gradiente
2. Genera nuovi dati falsi
3. Calcola loss: il D dovrebbe essere ingannato (output → 1)
4. Backpropagazione e aggiornamento pesi

### Metriche di valutazione

Non esiste una metrica standard per le GAN, ma monitoriamo:
- Loss del discriminatore e generatore
- Qualità visiva delle immagini generate
- Inception Score o FID per valutazioni più oggettive

### Esempio di output durante il training
- Epoch 1: Immagini rumore grigio
- Epoch 10: Forme vagamente numeriche
- Epoch 50: Cifre riconoscibili ma sfocate
- Epoch 100: Cifre ben definite che imitano lo stile MNIST

La bellezza delle GAN risiede in questo processo competitivo che porta entrambe le reti a migliorarsi reciprocamente, creando alla fine risultati sorprendentemente realistici.