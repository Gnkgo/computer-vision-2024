import numpy as np

from scipy import signal #for the scipy.signal.convolve2d function
from scipy import ndimage #for the scipy.ndimage.maximum_filter
import cv2
import matplotlib.pyplot as plt
# Harris corner detector
def extract_harris(img, sigma = 0.5, k = 0.04, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0
    
    # 1. Compute image gradients in x and y direction
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.
    #print("Hello World!")
    #raise NotImplementedError
        # I_x = I (x + 1, y) - I(x - 1, y) / 2
    # I_y = I (x, y + 1) - I(x, y - 1) / 2
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    I_x = signal.convolve2d(img, sobel_x, mode='same')
    I_y = signal.convolve2d(img, sobel_y, mode='same')
    
        # Plot the original image and the convolved images
    plt.figure(figsize=(12, 6))

    # plt.subplot(1, 3, 1)
    # plt.imshow(img, cmap='gray')
    # plt.title('Original Image')
    # plt.axis('off')

    # plt.subplot(1, 3, 2)
    # plt.imshow(I_x, cmap='gray')
    # plt.title('Sobel X')
    # plt.axis('off')

    # plt.subplot(1, 3, 3)
    # plt.imshow(I_y, cmap='gray')
    # plt.title('Sobel Y')
    # plt.axis('off')

    # plt.tight_layout()
    # plt.show()
    
    # 2. (Optional) Blur the computed gradients
    # TODO: compute the blurred image gradients
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)
    I_x = cv2.GaussianBlur(I_x, (5, 5), sigma, 0, cv2.BORDER_REPLICATE)
    I_y = cv2.GaussianBlur(I_y, (5, 5), sigma, 0, cv2.BORDER_REPLICATE)

    # 3. Compute elements of the local auto-correlation matrix "M"
    # TODO: compute the auto-correlation matrix here
    # You may refer to cv2.GaussianBlur or scipy.signal.convolve2d to perform the weighted sum
    I_x_2 = I_x * I_x
    I_y_2 = I_y * I_y
    I_x_y = I_x * I_y
    
    I_x_2 = cv2.GaussianBlur(I_x_2, (5, 5), sigma, 0)
    I_y_2 = cv2.GaussianBlur(I_y_2, (5, 5), sigma, 0)
    I_x_y = cv2.GaussianBlur(I_x_y, (5, 5), sigma, 0)
    
    
    # 4. Compute Harris response function C
    # TODO: compute the Harris response function C here
    #raise NotImplementedError
    det_M = I_x_2 * I_y_2 - I_x_y ** 2
    trace_M = I_x_2 + I_y_2
    C = det_M - k * (trace_M ** 2)
    
    # 5. Detection with threshold and non-maximum suppression
    # TODO: detection and find the corners here
    # For the non-maximum suppression, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    # You may refer to np.where to find coordinates of points that fulfill some condition; Please, pay attention to the order of the coordinates.
    # You may refer to np.stack to stack the coordinates to the correct output format
    #raise NotImplementedError
    
    corner_candidates = C > thresh

    C_max = ndimage.maximum_filter(C, size=(3, 3))

    corner_max = (C == C_max)
    final_corners = np.logical_and(corner_candidates, corner_max)
    corners = np.stack(np.where(final_corners)[::-1], axis=-1)

    return corners, C
    

