import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
import os
import json

class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader, lr, wd, epochs, device, framework='pytorch'):
        self.epochs = epochs
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.framework = framework
        if framework == 'pytorch':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            self.criterion = nn.CrossEntropyLoss()
        else:  # tensorflow
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr, weight_decay=wd),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

    def train(self, save=False, plot=False):
        if self.framework == 'pytorch':
            self.model.train()
            self.train_acc = []
            self.train_loss = []
            for epoch in range(self.epochs):
                total_loss = 0
                total_correct = 0
                total_samples = 0

                progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False)

                for batch in progress_bar:
                    input_datas, labels = batch
                    input_datas, labels = input_datas.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(input_datas)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    _, preds = outputs.max(1)
                    correct = (preds == labels).sum().item()
                    total = labels.size(0)

                    total_correct += correct
                    total_samples += total
                    total_loss += loss.item()

                    batch_accuracy = 100.0 * correct / total
                    average_accuracy = 100.0 * total_correct / total_samples
                    average_loss = total_loss / total_samples

                    progress_bar.set_postfix({
                        'Batch Acc': f'{batch_accuracy:.2f}%',
                        'Avg Acc': f'{average_accuracy:.2f}%',
                        'Loss': f'{average_loss:.4f}'
                    })
                self.train_acc.append(average_accuracy)
                self.train_loss.append(average_loss)
            if save:
                torch.save(self.model.state_dict(), "models/Abou_Birane_model.torch")
                print("Modèle sauvegardé sous models/Abou_Birane_model.torch")
            if plot:
                self.plot_training_history(save_path="models/pytorch_training_plot.png")
        else:  # tensorflow
            history = self.model.fit(
                self.train_dataloader,
                epochs=self.epochs,
                validation_data=self.test_dataloader,
                verbose=1
            )
            if save:
                project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                model_path = os.path.join(project_dir, "models", "Abou_Birane_model.tensorflow")
                print(f"Sauvegarde du modèle TensorFlow dans : {model_path}")
                self.model.save(model_path)
                print(f"Modèle sauvegardé sous {model_path}")
            if plot:
                plt.figure(figsize=(8, 5))
                plt.plot(history.history['accuracy'], color='tab:red', label='Accuracy')
                plt.plot(history.history['loss'], color='tab:blue', label='Loss')
                plt.title('Training Loss and Accuracy (TensorFlow)')
                plt.xlabel('Epoch')
                plt.ylabel('Value')
                plt.legend()
                plt.savefig("models/tensorflow_training_plot.png")
                plt.close()
                print("Figure sauvegardée sous models/tensorflow_training_plot.png")

    @torch.no_grad()
    def evaluate(self):
        if self.framework == 'pytorch':
            self.model.eval()
            total_loss = 0
            total_correct = 0
            total_samples = 0
            all_preds = []
            all_labels = []

            for inputs, labels in tqdm(self.test_dataloader, desc="Evaluating", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                _, preds = outputs.max(1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                total_loss += loss.item() * labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            avg_loss = total_loss / total_samples
            accuracy = 100.0 * total_correct / total_samples
            cm = confusion_matrix(all_labels, all_preds)
            print(f"\nTest Accuracy: {accuracy:.2f}%  |  Test Loss: {avg_loss:.4f}")
            print("Matrice de confusion :\n", cm)

            # Sauvegarder les métriques
            metrics = {
                "accuracy": accuracy,
                "loss": avg_loss,
                "confusion_matrix": cm.tolist()
            }
            with open("models/Abou_Birane_pytorch_metrics.json", "w") as f:
                json.dump(metrics, f)
            print("Métriques PyTorch sauvegardées sous models/Abou_Birane_pytorch_metrics.json")

            return accuracy, avg_loss
        else:  # tensorflow
            all_preds = []
            all_labels = []
            total_loss = 0
            total_samples = 0
            for inputs, labels in self.test_dataloader:
                preds = self.model.predict(inputs, verbose=0)
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, preds).numpy().mean()
                all_preds.extend(np.argmax(preds, axis=1))
                all_labels.extend(labels.numpy())
                total_loss += loss * labels.shape[0]
                total_samples += labels.shape[0]
            accuracy = 100.0 * accuracy_score(all_labels, all_preds)
            avg_loss = total_loss / total_samples
            cm = confusion_matrix(all_labels, all_preds)
            print(f"\nTest Accuracy: {accuracy:.2f}%  |  Test Loss: {avg_loss:.4f}")
            print("Matrice de confusion :\n", cm)

            # Sauvegarder les métriques
            metrics = {
                "accuracy": accuracy,
                "loss": avg_loss,
                "confusion_matrix": cm.tolist()
            }
            with open("models/Abou_Birane_tensorflow_metrics.json", "w") as f:
                json.dump(metrics, f)
            print("Métriques TensorFlow sauvegardées sous models/Abou_Birane_tensorflow_metrics.json")

            return accuracy, avg_loss

    def plot_training_history(self, save_path=None):
        if self.framework == 'pytorch':
            epochs = range(1, len(self.train_loss) + 1)
            fig, ax1 = plt.subplots(figsize=(8, 5))
            color_loss = 'tab:blue'
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss', color=color_loss)
            ax1.plot(epochs, self.train_loss, color=color_loss, label='Loss')
            ax1.tick_params(axis='y', labelcolor=color_loss)
            ax2 = ax1.twinx()
            color_acc = 'tab:red'
            ax2.set_ylabel('Accuracy (%)', color=color_acc)
            ax2.plot(epochs, self.train_acc, color=color_acc, label='Accuracy')
            ax2.tick_params(axis='y', labelcolor=color_acc)
            plt.title('Training Loss and Accuracy')
            fig.tight_layout()
            if save_path:
                plt.savefig(save_path)
                plt.close()
                print(f"Figure sauvegardée sous {save_path}")
            else:
                plt.show()