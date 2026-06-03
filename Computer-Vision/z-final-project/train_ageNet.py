import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from helperT import get_img_dataloaders, test_cel
from age_net import AgeNet


def train_ageNet(base_dir='DATASET/'):
    # hyper-parameters
    EPOCH = 100
    TRAIN_LR = 0.001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0

    # data
    train_loader, val_loader, test_loader = get_img_dataloaders(base_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = AgeNet(num_classes=101).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=TRAIN_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_mae = 1e9
    best_path = os.path.join(os.path.dirname(__file__), 'best_age_net.pth')

    ages = torch.arange(0, 101).float().to(device)

    for epoch in range(EPOCH):
        model.train()
        running_loss = 0.0
        for i, (y, x) in enumerate(train_loader):
            x = x.to(device).float()
            y = y.to(device).long()

            outputs = model(x)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # validation
        model.eval()
        predsVal = []
        gtsVal = []
        with torch.no_grad():
            for i, (y, x) in enumerate(val_loader):
                x = x.to(device).float()
                y = y.to(device).long()
                outputs = model(x)
                probs = torch.softmax(outputs, dim=1)
                pred_age = (probs * ages).sum(dim=1)
                predsVal.append(pred_age.cpu().numpy())
                gtsVal.append(y.cpu().numpy())

        predsVal = np.concatenate(predsVal, axis=0)
        gtsVal = np.concatenate(gtsVal, axis=0)
        mae = np.mean(np.abs(predsVal - gtsVal))

        print(f'Epoch {epoch+1}/{EPOCH}  loss={running_loss:.4f}  val_mae={mae:.4f}')

        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), best_path)
            print('Saved best model to', best_path)

    print('=> training finished')
    return best_path, test_loader


if __name__ == '__main__':
    best_path, test_loader = train_ageNet('DATASET/')

    # Load best model and run test to produce predictions file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AgeNet(num_classes=101).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.to(device)

    # helperT.test_cel expects the model to be on cuda and will call .cuda() on inputs.
    # If running on CPU, test_cel will still produce outputs if model and inputs are on CPU.
    prediction = test_cel(model, test_loader, os.path.join(os.path.dirname(__file__), 'custom.txt'))
    print('Test results saved to custom.txt')
