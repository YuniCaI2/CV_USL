from PIL import Image
import numpy as np


class RGBdot_matching:
    def __init__(self,npy_path,target_rgb,alpha,bias):
        self.npy_path = npy_path
        self.target_rgb = target_rgb
        self.alpha = alpha
        self.bias = bias
    
    def GetWeightMatrix(self):
        return weight_mapping(self.npy_path,self.target_rgb,self.alpha,self.bias)
        
        


def find_color_coordinates(npy_path,target_rgb):
    feature_array = np.load(npy_path)
    coordinates = []
    for rgb in target_rgb:
        rgb = np.array(rgb)
        coordinate = np.argwhere(np.all(feature_array == rgb, axis=-1))
        coordinates.extend(coordinate)
    return coordinates
    
def compute_euclidean_distance_ratio(x1,x2,width):## 这里的x1和x2是代表两个点
    return ((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)**0.5/width

def weight_function(distance,alpha,bias):
    return (1/(np.exp(alpha*distance-bias)+1))## bias means the threshold of this function and it should be bigger than zero
    
def get_width(npy_path):
    image = np.load(npy_path)
    return image.shape[-2]

def get_height(npy_path):
    image = np.array(npy_path)
    return image.shape[-3]

def weight_mapping(npy_path, target_rgb, alpha, bias):
    coordinates = find_color_coordinates(npy_path, target_rgb)
    image_width = get_width(npy_path)
    image_height = get_height(npy_path)
    
    # Initialize the weight matrix
    weight_matrix = np.zeros((image_height, image_width))
    
    # Load the extracted image
    
    num_coordinates = len(coordinates)
    if num_coordinates == 0:
        return weight_matrix  # No target color found, return an empty weight matrix
    
    # Iterate through each pixel in the image
    for y in range(image_height):
        for x in range(image_width):
            pixel = (x, y)
            # Calculate the sum of weights from all target color coordinates
            total_weight = 0
            for coord in coordinates:
                distance_ratio = compute_euclidean_distance_ratio(pixel, coord, image_width)
                total_weight += weight_function(distance_ratio, alpha, bias)
            
            # Compute the average weight for the pixel
            average_weight = total_weight / num_coordinates
            weight_matrix[y, x] = average_weight
    
    return weight_matrix   
        
    
