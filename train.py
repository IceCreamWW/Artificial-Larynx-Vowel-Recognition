import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import time
import copy
import argparse
import logging



class Finetune:
    def __init__(self, data_root, expdir, model_name, device):
        self.data_root = data_root
        data_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.208, 0.861, 0.773], [0.236, 0.167, 0.246])
        ])

        self.datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_root, x), data_transform) for x in ['train', 'valid']}
        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'valid']}
        self.model = self.build_model(model_name).to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        self.criterion = nn.CrossEntropyLoss()
        self.expdir = expdir
        self.device = device

    def build_model(self, model_name):
        build_model_func = torchvision.models.__dict__[model_name]
        model = build_model_func(pretrained=True)

        if model_name.startswith("resnet"):
            model.fc = nn.Linear(model.fc.in_features, 5)
        elif model_name.startswith("densenet"):
            model.classifier = nn.Linear(model.classifier.in_features, 5)
        else:
            raise NotImplementedError(f"finetuning model ${model_name} is not supported")
        return model


    def train_one_epoch(self):
        self.model.train()
        epoch_loss = epoch_correct = 0

        for batch in self.dataloaders['train']:
            inputs, labels = [item.to(self.device) for item in batch]

            self.optimizer.zero_grad()
            logits = self.model(inputs)
            preds = logits.argmax(dim=1)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            epoch_correct += (preds == labels.detach()).sum()

        self.scheduler.step()
        epoch_loss = epoch_loss / len(self.datasets['train'])
        epoch_acc = epoch_correct /  len(self.datasets['train'])
        return epoch_acc, epoch_loss

    def valid_one_epoch(self):
        self.model.eval()
        epoch_loss = epoch_correct = 0

        for batch in self.dataloaders['valid']:
            inputs, labels = [item.to(self.device) for item in batch]

            with torch.no_grad():
                logits = self.model(inputs)
                preds = logits.argmax(dim=1)
                loss = self.criterion(logits, labels)

            epoch_loss += loss.item() * inputs.size(0)
            epoch_correct += (preds == labels.detach()).sum()

        epoch_loss = epoch_loss / len(self.datasets['valid'])
        epoch_acc = epoch_correct /  len(self.datasets['valid'])
        return epoch_acc, epoch_loss


    def __call__(self, max_epochs=20):
        start = time.time()
        best_valid_acc = -1
        best_model_wts = None
        assert max_epochs > 0, f"max epochs = {max_epochs} <= 0" 
        for iepoch in range(1, max_epochs + 1):
            logging.info(f'epoch {iepoch} / {max_epochs}')
            epoch_train_acc, epoch_train_loss = self.train_one_epoch()
            logging.info(f'train loss: {epoch_train_loss:.4f} acc: {epoch_train_acc:.4f}')

            epoch_valid_acc, epoch_valid_loss = self.valid_one_epoch()
            logging.info(f'valid loss: {epoch_valid_loss:.4f} acc: {epoch_valid_acc:.4f}')

            if epoch_valid_acc > best_valid_acc:
                best_valid_acc = epoch_valid_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())

        time_elapsed = time.time() - start
        logging.info(f'train complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        logging.info(f'last valid acc: {epoch_valid_acc:.4f}')
        logging.info(f'best valid acc: {best_valid_acc:.4f}')

        last_ckpt_path = os.path.join(self.expdir, "last.pth") 
        best_ckpt_path = os.path.join(self.expdir, "valid.acc.best.pth") 
        torch.save(self.model.state_dict(), last_ckpt_path)
        logging.info(f"last model saved to {last_ckpt_path}")
        torch.save(best_model_wts, best_ckpt_path)
        logging.info(f"best model saved to {best_ckpt_path}")

        return self.model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="finetune pretrained model for vowel recognition")
    parser.add_argument("--data-root", type=str, required=True, help="path to data dir")
    parser.add_argument("--expdir", type=str, required=True, help="path to save model and logs")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18","densenet161"])
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(filename=os.path.join(args.expdir, "train.log"), format='%(asctime)s: %(message)s', level=logging.INFO)

    # try to be deterministic
    logging.info(f"seed = {args.seed}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    finetune = Finetune(data_root=args.data_root, expdir=args.expdir, model_name=args.model, device=device)
    finetune(max_epochs=args.max_epochs)

