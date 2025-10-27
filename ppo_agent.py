import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


# Sürekli aksiyon uzayları için bir Aktör (Politika) Ağı
# Bu ağ, belirli bir durum (state) için hangi eylemin (action)
# yapılacağına dair bir olasılık dağılımı (ortalama ve standart sapma) üretir.
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorNetwork, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.mean_layer = nn.Linear(256, action_dim)  # Her eylem için bir ortalama değeri
        self.log_std_layer = nn.Linear(256, action_dim)  # Her eylem için bir log(standart sapma) değeri

        # PPO'da genellikle standart sapma da öğrenilir.
        # Bu, ajanın keşif (exploration) seviyesini ayarlamasını sağlar.

        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.layer_1(state))
        x = torch.relu(self.layer_2(x))

        # Ortalama (mean) değerini -max_action ile +max_action arasında olacak şekilde ölçeklendir.
        # Tanh fonksiyonu çıktıyı (-1, 1) arasına sıkıştırır.
        mean = self.max_action * torch.tanh(self.mean_layer(x))

        # Log standart sapma, genellikle belirli bir aralıkta tutulur.
        # Bu, standart sapmanın çok büyümesini veya küçülmesini engeller.
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)

        # Bu ortalama ve standart sapma ile bir Normal (Gauss) dağılımı oluşturulur.
        # Eğitim sırasında bu dağılımdan eylemler örneklenir.
        dist = Normal(mean, std)

        return dist


# Kritik (Değer) Ağı
# Bu ağ, belirli bir durumun (state) ne kadar "iyi" olduğunu tahmin eder.
# Yani, o durumdan itibaren beklenen toplam ödülün ne olacağını söyler.
class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.value_layer = nn.Linear(256, 1)  # Çıktı tek bir değer: V(s)

    def forward(self, state):
        x = torch.relu(self.layer_1(state))
        x = torch.relu(self.layer_2(x))
        value = self.value_layer(x)
        return value