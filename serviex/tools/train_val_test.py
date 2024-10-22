import torch
import wandb
import json
import os
from tools.utils import train_and_validate, test, classify, calculate_metrics, convert_to_serializable


def run(exp_config, train_dataloader, val_dataloader, test_dataloader, model, criterion, optimizer, num_epochs=15):

    model_name: str = model.__class__.__name__

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project="CNN_CatsvsDogs", entity="ar-um", tags=["BERTOLDI_MANCUSO"], name="Bertoldi_Mancuso_" + model_name)
    wandb.config.update(exp_config)

    model = model.to(device)
    exp_config['model'] = model_name

    early_stopping_patience = 5
    exp_config['num_epochs'] = num_epochs
    exp_config['early_stopping_patience'] = early_stopping_patience

    base_checkpoint_path = '/checkpoints'

    os.makedirs(base_checkpoint_path, exist_ok=True)

    checkpoint_path = base_checkpoint_path + '/best_model' + model_name + '.pth'

    train_and_validate(model, train_dataloader, val_dataloader, criterion, optimizer, device,
                       num_epochs, early_stopping_patience, checkpoint_path)

    model.eval()

    y_true, y_proba = test(model, test_dataloader, device)

    y_true, y_pred, y_proba_flat = classify(y_proba, y_true)

    accuracy, precision, recall, specificity, fpr, tpr, roc_auc = calculate_metrics(y_true, y_pred, y_proba_flat)

    roc_data = [[x, y] for (x, y) in zip(fpr, tpr)]
    table = wandb.Table(data=roc_data, columns=["FPR", "TPR"])
    wandb.log({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_specificity": specificity,
        "test_confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_true.flatten().tolist(),
            preds=y_pred.flatten().tolist(),
            class_names=["Clase 0", "Clase 1"]
        ),
        "ROC Curve": wandb.plot.line(table, "FPR", "TPR", title="ROC Curve"),
        "test_roc_auc": roc_auc,
    })

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Specificity: {specificity:.2f}")

    serializable_exp_config = convert_to_serializable(exp_config)

    dicts_path = os.path.join(os.path.expanduser("~"), 'dicts')
    os.makedirs(dicts_path, exist_ok=True)
    with open(dicts_path + '/exp_config' + model_name + '.json', 'w') as json_file:
        json.dump(serializable_exp_config, json_file, indent=4)

    wandb.finish()

    return exp_config
