import numpy as np
import cv2
import glob
import os
from sklearn.cluster import KMeans
from tqdm import tqdm
from bow import findnn, grid_points, create_bow_histograms, create_codebook, bow_recognition_nearest
# Your existing functions here...

# Additional imports for file handling
import itertools

def run_experiment(nameDirPos_train, nameDirNeg_train, nameDirPos_test, nameDirNeg_test, k_values, numiter_values):
    # Open the file once at the beginning
    with open('accuracies.txt', 'w') as f:
        f.write('k,numiter,acc_pos,acc_neg\n')  # Write header

        for k, numiter in itertools.product(k_values, numiter_values):
            print(f'Running with k={k} and numiter={numiter}')
            
            # Create codebook
            vCenters = create_codebook(nameDirPos_train, nameDirNeg_train, k, numiter)

            # Create bag-of-words histograms
            vBoWPos = create_bow_histograms(nameDirPos_train, vCenters)
            vBoWNeg = create_bow_histograms(nameDirNeg_train, vCenters)

            # Test positive samples
            vBoWPos_test = create_bow_histograms(nameDirPos_test, vCenters)
            result_pos = 0
            for i in range(vBoWPos_test.shape[0]):
                cur_label = bow_recognition_nearest(vBoWPos_test[i:(i + 1)], vBoWPos, vBoWNeg)
                result_pos += cur_label
            acc_pos = result_pos / vBoWPos_test.shape[0]
            
            # Test negative samples
            vBoWNeg_test = create_bow_histograms(nameDirNeg_test, vCenters)
            result_neg = 0
            for i in range(vBoWNeg_test.shape[0]):
                cur_label = bow_recognition_nearest(vBoWNeg_test[i:(i + 1)], vBoWPos, vBoWNeg)
                result_neg += cur_label
            acc_neg = 1 - result_neg / vBoWNeg_test.shape[0]

            # Write results to the file immediately after calculation
            f.write(f'{k},{numiter},{acc_pos},{acc_neg}\n')
            f.flush()  # Optional: Ensures data is written to the file immediately

if __name__ == '__main__':
    # set a fixed random seed
    np.random.seed(42)

    # Directories for training and testing data
    nameDirPos_train = 'data/data_bow/cars-training-pos'
    nameDirNeg_train = 'data/data_bow/cars-training-neg'
    nameDirPos_test = 'data/data_bow/cars-testing-pos'
    nameDirNeg_test = 'data/data_bow/cars-testing-neg'

    # Define ranges for k and numiter
    k_values = range(10, 51, 5)  # Example values for k
    numiter_values = range(10, 101, 10)  # Example values for numiter

    # Run the experiments
    run_experiment(nameDirPos_train, nameDirNeg_train, nameDirPos_test, nameDirNeg_test, k_values, numiter_values)
