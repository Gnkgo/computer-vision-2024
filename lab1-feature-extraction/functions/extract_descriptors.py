import numpy as np
import cv2
def filter_keypoints(img, keypoints, patch_size=9):
    '''
    Inputs:
    - img:         Original image (H, W, C)
    - keypoints:   (q, 2) numpy array of keypoint locations [x, y]
    - patch_size:  Size of the patches to extract (default is 9x9)
    
    Returns:
    - filtered_keypoints: (q', 2) numpy array of filtered keypoint locations [x, y]
    '''
    # Calculate the half patch size to avoid boundary issues
    half_patch = patch_size // 2
    
    # Get the dimensions of the image
    h, w = img.shape[:2]
    
    # Filter out keypoints that are too close to the image boundaries
    filtered_keypoints = []
    for kp in keypoints:
        x, y = kp
        if x > half_patch and x < (w - half_patch) and y > half_patch and y < (h - half_patch):
            filtered_keypoints.append(kp)
    
    # Convert back to a numpy array
    filtered_keypoints = np.array(filtered_keypoints)
    
    return filtered_keypoints
        

# The implementation of the patch extraction is already provided here
def extract_patches(img, keypoints, patch_size = 9):
    '''
    Extract local patches for each keypoint
    Inputs:
    - img:          (h, w) gray-scaled images
    - keypoints:    (q, 2) numpy array of keypoint locations [x, y]
    - patch_size:   size of each patch (with each keypoint as its center)
    Returns:
    - desc:         (q, patch_size * patch_size) numpy array. patch descriptors for each keypoint
    '''
    h, w = img.shape[0], img.shape[1]
    img = img.astype(float) / 255.0
    offset = int(np.floor(patch_size / 2.0))
    ranges = np.arange(-offset, offset + 1)
    desc = np.take(img, ranges[:,None] * w + ranges + (keypoints[:, 1] * w + keypoints[:, 0])[:, None, None]) # (q, patch_size, patch_size)
    desc = desc.reshape(keypoints.shape[0], -1) # (q, patch_size * patch_size)
    return desc

