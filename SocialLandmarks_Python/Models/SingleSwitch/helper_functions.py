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
    m = (matches1[idx] == True).sum().item()
    return m