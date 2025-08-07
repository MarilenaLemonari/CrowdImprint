from imports import *

def scale_to_standard_normal(images):
    mean = np.mean(images) 
    std = np.std(images) 
    scaled_images = (images - mean) / std
    return scaled_images

def make_cm(model, x, y, name):
    y_pred = model.predict(x)
    y_pred_classes = y_pred.argmax(axis=-1)
    confusion = confusion_matrix(y, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=False, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{name} Confusion Matrix')
    plt.savefig(f'{name}_confusion_matrix.png')

    return confusion