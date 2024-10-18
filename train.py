import torch
from utils.utils import load_config
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, recall_score, precision_score

config = load_config('configs/configs.yaml')

def train_multimodal(
        log_interval, 
        modalities,
        model, 
        device, 
        train_loader, 
        criterion, 
        optimizer, 
        epoch):

    model.train()

    losses = []
    scores = []
    all_y = []
    all_y_pred = []
    N_count = 0
    for batch_idx, (X, y) in enumerate(train_loader):
        for modality in modalities:
            X[modality+'_features'] = X[modality+'_features'].to(device)
        y = y.to(device).view(-1, )
        N_count += X[modality+'_features'].size(0)

        optimizer.zero_grad()
        output = model(X, mode='train')

        loss = criterion(output, y)
        losses.append(loss.item())

        y_pred = output.max(1, keepdim=True)[1]
        all_y.extend(y)
        all_y_pred.extend(y_pred)

        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            metrics = evalMetric(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())


    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    scores = evalMetric(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    loss = sum(losses) / len(losses)

    return loss, scores

def evalMetric(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    mf1Score = f1_score(y_true, y_pred, average='macro')
    f1Score  = f1_score(y_true, y_pred, labels = np.unique(y_pred))
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    area_under_c = auc(fpr, tpr)
    recallScore = recall_score(y_true, y_pred, labels = np.unique(y_pred))
    precisionScore = precision_score(y_true, y_pred, labels = np.unique(y_pred))
    return dict({"accuracy": accuracy, 'mF1Score': mf1Score, 'f1Score': f1Score, 'auc': area_under_c,
           'precision': precisionScore, 'recall': recallScore})

def validation_multimodal(
        modalities, 
        model, 
        device, 
        criterion, 
        test_loader, 
        dataset_name):

    model.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            for modality in modalities:
                X[modality+'_features'] = X[modality+'_features'].to(device)
            y = y.to(device).view(-1, )

            output = model(X)

            loss = criterion(output, y)
            test_loss += loss.item()
            y_pred = output.max(1, keepdim=True)[1]
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    metrics = evalMetric(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())


    return test_loss, metrics, list(all_y_pred.cpu().data.squeeze().numpy())
