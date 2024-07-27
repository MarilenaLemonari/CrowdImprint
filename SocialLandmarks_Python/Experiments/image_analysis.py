from imports import *
import tifffile
from CNNPytorch import *

# cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Experiments
folder_path = "C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Experiments"
tiff_files = [f for f in os.listdir(folder_path) if f.endswith('.tiff') or f.endswith('.tif')]
    
# Load all TIFF images
images = []
for file_name in tiff_files:
    file_path = os.path.join(folder_path, file_name)
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    images.append(image)
    print(file_name)
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)  
    # cv2.destroyAllWindows()
    # image[np.where(image != np.max(image))] = 0 
    # tifffile.imwrite(f'{file_name}.tif', image)

images_array = np.array(images)
images_array =  images_array[:,np.newaxis, :,:]
# tifffile.imwrite('img.tif', img)

model = CNN()
model.load_state_dict(torch.load('C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\\model_canoisy_new.pth'))

model.eval()

inputs = torch.tensor(images_array, dtype = torch.float32)# torch.stack((torch.Tensor(image1),torch.Tensor(image2))).unsqueeze(1)
print(inputs.shape)

with torch.no_grad():
    output = model(inputs)
    print(output.shape)

print(np.argmax(output[0,:]), np.argmax(output[1,:]), np.argmax(output[2,:]), np.argmax(output[3,:]), np.argmax(output[4,:]))

index_list = [0,1,2,3,4]
for index in index_list:
    sorted_indices = sorted(range(len(output[index,:])), key=lambda i: output[index,i], reverse=True)
    print("Index ", index, ": ",sorted_indices)
