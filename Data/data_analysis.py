# IMPORTS:
import numpy as np
from tqdm import tqdm

# INSTRUCTIONS:
#   cd C:\PROJECTS\SocialLandmarks
#   .venv/Scripts/activate
#   cd .\Data\
#   python3 .\data_analysis.py

def preprocess_data():
  # array of shape [number of characters, timesteps, 2 (x,y)]
  print("preprocess_data()")

  array = np.zeros((1,5,2))
  array[0] = np.array([[0,0],[1,1],[2,2],[3,3],[4,4]])
  return array


if __name__ ==  '__main__':

  print("Main")
  array = preprocess_data()
  
