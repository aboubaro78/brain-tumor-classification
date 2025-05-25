import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from tensorflow.keras import layers, models

class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, padding=2)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(20 * 56 * 56, 50)
        self.fc2 = nn.Linear(50, 4)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 20 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def create_tf_model():
    model = models.Sequential([
        layers.Conv2D(10, (5, 5), padding='same', activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(20, (5, 5), padding='same', activation='relu'),
        layers.Dropout(0.5),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(50, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')
    ])
    return model