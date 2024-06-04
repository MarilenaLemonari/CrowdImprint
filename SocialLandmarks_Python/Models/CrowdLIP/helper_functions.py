from imports import *

def visualize_image(array):
    image_rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    # plt.imshow(array, cmap='gray')
    plt.title('Image')
    plt.axis('off')
    plt.show()

def scale_to_standard_normal(images):
    mean = np.mean(images) 
    std = np.std(images) 
    scaled_images = (images - mean) / std
    return scaled_images

def accuracy_f(y_true, y_prob, correct, total):
    # Find y_pred:
    _, y_pred = torch.max(y_prob, 1)

    y_pred_r = y_pred 

    matches = (y_pred_r == y_true)
    correct += matches.sum().item()
    total += y_true.size(0)

    return correct, total, matches

def accuracy_overall(matches1, matches2):
    idx = [i for i in range(len(matches1)) if matches1[i] == matches2[i]]
    m = (matches1[idx] == True)

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

def contrastive_loss(y_true, y_pred):
    margin = 1.0
    y_true = tf.cast(y_true, y_pred.dtype)
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def euclidean_distance(vectors):
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def compute_similarity(img1, img2):
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)
    return model.predict([img1, img2])[0]

def accuracy(y_true, y_pred, threshold=0.5):
    y_pred_label = K.cast(y_pred < threshold, y_true.dtype)
    return K.mean(K.equal(y_true, y_pred_label))