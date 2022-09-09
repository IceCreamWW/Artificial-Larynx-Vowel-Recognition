import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import time
import copy
import argparse
import logging

class Inference:
    def __init__(self, data_root, model_name, ckpt_path, device):
        self.data_root = data_root
        data_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.208, 0.861, 0.773], [0.236, 0.167, 0.246])
        ])

        self.dataset = torchvision.datasets.ImageFolder(os.path.join(data_root, "valid"), data_transform)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=100, shuffle=False, num_workers=4)
        self.model = self.build_model(model_name).to(device)
        self.model.load_state_dict(torch.load(ckpt_path))
        self.device = device

    def build_model(self, model_name):
        build_model_func = torchvision.models.__dict__[model_name]
        model = build_model_func(pretrained=False)

        if model_name.startswith("resnet"):
            model.fc = nn.Linear(model.fc.in_features, 5)
        elif model_name.startswith("densenet"):
            model.classifier = nn.Linear(model.classifier.in_features, 5)
        else:
            raise NotImplementedError(f"finetuning model ${model_name} is not supported")
        return model


    def __call__(self):
        self.model.eval()
        correct = 0

        for batch in self.dataloader:
            inputs, labels = [item.to(self.device) for item in batch]

            with torch.no_grad():
                logits = self.model(inputs)
                preds = logits.argmax(dim=1)

            correct += (preds == labels.detach()).sum()

        acc = correct /  len(self.dataset)
        return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test model for vowel recognition")
    parser.add_argument("--data-root", type=str, required=True, help="path to data dir")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18","densenet161"])
    parser.add_argument("--ckpt-path", type=str, default="resnet18", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference = Inference(data_root=args.data_root, model_name=args.model, ckpt_path=args.ckpt_path, device=device)
    acc = inference()
    print(f"Acc: {acc:.4f}")

