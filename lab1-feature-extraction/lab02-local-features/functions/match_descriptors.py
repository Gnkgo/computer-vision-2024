import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please
    #raise NotImplementedError
    return np.sum(desc1 ** 2, axis=1)[:, None] + np.sum(desc2 ** 2, axis=1) - 2 * np.dot(desc1, desc2.T)

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here
        # You may refer to np.argmin to find the index of the minimum over any axis
        print(distances, "distances")
        nearest_neighbors_indices = np.argmin(distances, axis=1)
        print(q1, "q1")
        print(nearest_neighbors_indices, "nearest_neighbors_indices")
        print(np.arange(q1), "np.arange(q1)")
        matches = np.column_stack((np.arange(q1), nearest_neighbors_indices))
        #raise NotImplementedError
        
        
        #raise NotImplementedError
    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        # You may refer to np.min to find the minimum over any axis
        match1 = np.argmin(distances, axis=1)
        distances2 = ssd(desc2, desc1)
        match2 = np.argmin(distances2, axis=1)
        matches = np.column_stack((np.where(match2[match1] == np.arange(q1))[0], match1[np.where(match2[match1] == np.arange(q1))]))
        
    elif method == "ratio":
        nearest_neighbors_indices = np.argmin(distances, axis=1)
        sorted_distances = np.partition(distances, 2, axis=1)
        nearest_distances = sorted_distances[:, 0]  
        second_nearest_distances = sorted_distances[:, 1] 
        ratio = nearest_distances / second_nearest_distances
        good_matches_mask = ratio < ratio_thresh
        
        good_matches_indices = np.arange(q1)[good_matches_mask]
        good_nearest_neighbors = nearest_neighbors_indices[good_matches_mask]
        matches = np.column_stack((good_matches_indices, good_nearest_neighbors))
        
        return matches
    else:
        raise NotImplementedError
    return matches

