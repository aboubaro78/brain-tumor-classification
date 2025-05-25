import torch
import argparse
from utils.prep import get_data, get_tensorflow_datasets
from models.cnn import CNN1, create_tf_model
from models.train import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement d'un modèle CNN")
    parser.add_argument('--model', type=str, choices=['pytorch', 'tensorflow'], default='pytorch', help="Type de modèle (pytorch ou tensorflow)")
    parser.add_argument('--train_dir', type=str, default='data/training', help="Chemin vers les données d'entraînement")
    parser.add_argument('--test_dir', type=str, default='data/testing', help="Chemin vers les données de test")
    parser.add_argument('--epochs', type=int, default=10, help="Nombre d'époques d'entraînement")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--wd', type=float, default=0.0001, help="Weight decay")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train', help="Mode: train ou eval")
    parser.add_argument('--cuda', action='store_true', help="Utiliser le GPU si disponible")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Utilisation de {device}")

    if args.model == 'pytorch':
        train_dataloader, test_dataloader = get_data(args.train_dir, args.test_dir)
        model = CNN1().to(device)
        if args.mode == 'eval':
            model.load_state_dict(torch.load("models/Abou_Birane_model.torch"))
        trainer = Trainer(model, train_dataloader, test_dataloader, args.lr, args.wd, args.epochs, device, framework='pytorch')
    else:  # tensorflow
        train_dataloader, test_dataloader = get_tensorflow_datasets(args.train_dir, args.test_dir)
        model = create_tf_model()
        if args.mode == 'eval':
            model = tf.keras.models.load_model("models/Abou_Birane_model.tensorflow")
        trainer = Trainer(model, train_dataloader, test_dataloader, args.lr, args.wd, args.epochs, device, framework='tensorflow')

    if args.mode == 'train':
        trainer.train(save=True, plot=True)
    trainer.evaluate()

if __name__ == '__main__':
    main()