import torch
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_score, recall_score
import cv2


def train(model, train_dataloader, criterion, optimizer, device):
    model.to(device)
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        threshold = 0.5
        predicted = (outputs.detach() >= threshold)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_avg_loss = running_loss / len(train_dataloader)
    train_accuracy = correct / total

    return train_avg_loss, train_accuracy


def validate(model, val_dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            threshold = 0.5
            predicted = (outputs.detach() >= threshold)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_avg_loss = running_loss / len(val_dataloader)
    val_accuracy = correct / total

    return val_avg_loss, val_accuracy


def train_and_validate(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs,
                       early_stopping_patience, checkpoint_path):

    epochs_without_improvement = 0
    best_val_loss = 5

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_dataloader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_dataloader, criterion, device)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}, '
              f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}')

        wandb.log({"epochs": epoch,
                   "train_acc": train_accuracy,
                   "train_loss": train_loss,
                   "val_acc": val_accuracy,
                   "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            epochs_without_improvement = 0
            print("Checkpoint saved")

        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == early_stopping_patience:
                print("Early Stopping")
                break


def test(model, test_dataloader, device):
    y_true = []
    y_proba = []

    for image, label in test_dataloader:
        image, label = image.to(device), label.to(device)

        with torch.no_grad():
            output = model(image)

            y_true.append(label.to("cpu").float())
            y_proba.append(output.to("cpu").float())

    return y_true, y_proba


def classify(y_proba, y_true, thr=0.5):

    y_true_tensor = torch.cat(y_true)
    y_proba_tensor = torch.cat(y_proba)

    y_pred_tensor = (y_proba_tensor >= thr).int()

    y_true = y_true_tensor.numpy()
    y_pred = y_pred_tensor.numpy()

    y_proba_flat = y_proba_tensor.numpy().ravel()

    return y_true, y_pred, y_proba_flat


def calculate_metrics(y_true, y_pred, y_proba_flat):
    if len(y_proba_flat.shape) == 3:
        y_proba_flat = y_proba_flat[:, :, 1].flatten()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    specificity = recall_score(y_true, y_pred, pos_label=0)

    fpr, tpr, _ = roc_curve(y_true, y_proba_flat)
    roc_auc = auc(fpr, tpr)

    return accuracy, precision, recall, specificity, fpr, tpr, roc_auc


def convert_to_serializable(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(x) for x in obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    return obj




# Load the uploaded images
image_front_path = '/mnt/data/imagen_1_frente.jpeg'
image_back_path = '/mnt/data/imagen_1_dorso.jpeg'

image_front = cv2.imread(image_front_path)
image_back = cv2.imread(image_back_path)


def make_image_look_printed(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a GaussianBlur to simulate slight ink spread or imperfection
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    # Add paper texture using noise
    noise = np.random.normal(0, 25, gray.shape).astype(np.uint8)
    noisy_image = cv2.add(blurred, noise)

    # Add a slight yellow tint to simulate paper aging
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])

    sepia_image = cv2.transform(cv2.cvtColor(noisy_image, cv2.COLOR_GRAY2BGR), sepia_filter)
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)

    return sepia_image
