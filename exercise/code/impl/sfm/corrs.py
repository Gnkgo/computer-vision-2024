import numpy as np

# Find (unique) 2D-3D correspondences from 2D-2D correspondences
def Find2D3DCorrespondences(image_name, images, matches, registered_images):
  assert(image_name not in registered_images)

  image_kp_idxs = []
  p3D_idxs = []
  for other_image_name in registered_images:
    other_image = images[other_image_name]
    pair_matches = GetPairMatches(image_name, other_image_name, matches)

    for i in range(pair_matches.shape[0]):
      p3D_idx = other_image.GetPoint3DIdx(pair_matches[i,1])
      if p3D_idx > -1:
        p3D_idxs.append(p3D_idx)
        image_kp_idxs.append(pair_matches[i,0])

  print(f'found {len(p3D_idxs)} points, {np.unique(np.array(p3D_idxs)).shape[0]} unique points')

  # Remove duplicated correspondences
  _, unique_idxs = np.unique(np.array(p3D_idxs), return_index=True)
  image_kp_idxs = np.array(image_kp_idxs)[unique_idxs].tolist()
  p3D_idxs = np.array(p3D_idxs)[unique_idxs].tolist()
  
  return image_kp_idxs, p3D_idxs


# Make sure we get keypoint matches between the images in the order that we requested
def GetPairMatches(im1, im2, matches):
  if im1 < im2:
    return matches[(im1, im2)]
  else:
    return np.flip(matches[(im2, im1)], 1)


def UpdateReconstructionState(new_points3D, corrs, points3D, images):
    """
    Update the 3D reconstruction with newly triangulated points and their correspondences.

    Args:
        new_points3D (ndarray): Newly triangulated 3D points (N x 3).
        corrs (dict): Dictionary mapping image names to keypoint indices for the new points.
        points3D (ndarray): Existing global 3D points (M x 3).
        images (dict): Dictionary of image objects.

    Returns:
        points3D (ndarray): Updated 3D points array.
        images (dict): Updated image dictionary with new correspondences.
    """

    # Get the starting index for the new points in the global points3D array
    start_idx = points3D.shape[0]

    # Append the new 3D points to the global points array
    points3D = np.append(points3D, new_points3D, axis=0)

    # Iterate through the correspondences
    for im_name, kp_indices in corrs.items():
        # Calculate the global indices of the new points
        point_indices = np.arange(start_idx, start_idx + len(kp_indices))

        # Add the 2D-3D correspondences to the image
        images[im_name].Add3DCorrs(kp_indices, point_indices)

    return points3D, images
