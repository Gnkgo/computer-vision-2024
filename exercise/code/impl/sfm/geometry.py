import numpy as np

from impl.dlt import BuildProjectionConstraintMatrix
from impl.util import MakeHomogeneous, HNormalize
from impl.sfm.corrs import GetPairMatches
# from impl.opt import ImageResiduals, OptimizeProjectionMatrix

# # Debug
# import matplotlib.pyplot as plt
# from impl.vis import Plot3DPoints, PlotCamera, PlotProjectedPoints

# Placeholder for BuildProjectionConstraintMatrix and other imports.
def EstimateEssentialMatrix(K, im1, im2, matches):


    # Convert keypoints to homogeneous coordinates
    kp1_h = MakeHomogeneous(im1.kps, 1).T  # Shape (3, N)
    kp2_h = MakeHomogeneous(im2.kps, 1).T # Shape (3, N)
    
    # Normalize keypoints using the intrinsic matrix
    normalized_kps1 = (np.linalg.inv(K) @ kp1_h).T  # Shape (N, 3)
    normalized_kps2 = (np.linalg.inv(K) @ kp2_h).T  # Shape (N, 3)
    

    # Assemble the constraint matrix
    constraint_matrix = np.zeros((matches.shape[0], 9))
    for i in range(matches.shape[0]):
        x1, y1 = normalized_kps1[matches[i, 0], 0], normalized_kps1[matches[i, 0], 1]
        x2, y2 = normalized_kps2[matches[i, 1], 0], normalized_kps2[matches[i, 1], 1]

        constraint_matrix[i] = [
            x2 * x1, x2 * y1, x2,
            y2 * x1, y2 * y1, y2,
            x1, y1, 1
        ]

    # Solve for the nullspace of the constraint matrix
    _, _, vh = np.linalg.svd(constraint_matrix)
    vectorized_E_hat = vh[-1, :]

    # Reshape the vectorized matrix into its 3x3 form
    E_hat = vectorized_E_hat.reshape(3, 3)

    # Enforce the singularity constraints on E
    u, s, vt = np.linalg.svd(E_hat)
    s_corrected = [1, 1, 0]  # Two singular values are equal; the third is zero
    E = u @ np.diag(s_corrected) @ vt

    # This is just a quick test that should tell you if your estimated matrix is not correct
    # It might fail if you estimated E in the other direction (i.e. kp2' * E * kp1)
    # You can adapt it to your assumptions.
    for i in range(matches.shape[0]):
      kp1 = normalized_kps1[matches[i,0],:]
      kp2 = normalized_kps2[matches[i,1],:]
      # print("assertion error: ", abs(kp2 @ E @ kp1))
      error = abs(kp2.transpose() @ E @ kp1) 
      assert abs(kp2.transpose() @ E @ kp1) < 0.01, f"Estimated E does not satisfy the epipolar constraint: {error}"

    return E



def DecomposeEssentialMatrix(E):

  u, s, vh = np.linalg.svd(E)

  # Determine the translation up to sign
  t_hat = u[:,-1]

  W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
  ])

  # Compute the two possible rotations
  R1 = u @ W @ vh
  R2 = u @ W.transpose() @ vh

  # Make sure the orthogonal matrices are proper rotations (Determinant should be 1)
  if np.linalg.det(R1) < 0:
    R1 *= -1

  if np.linalg.det(R2) < 0:
    R2 *= -1

  # Assemble the four possible solutions
  sols = [
    (R1, t_hat),
    (R2, t_hat),
    (R1, -t_hat),
    (R2, -t_hat)
  ]

  return sols


def TriangulatePoints(K, im1, im2, matches):
  R1, t1 = im1.Pose()
  R2, t2 = im2.Pose()
  P1 = K @ np.append(R1, np.expand_dims(t1, 1), 1)
  P2 = K @ np.append(R2, np.expand_dims(t2, 1), 1)

  # Ignore matches that already have a triangulated point
  new_matches = np.zeros((0, 2), dtype=int)
  num_matches = matches.shape[0]
  for i in range(num_matches):
    p3d_idx1 = im1.GetPoint3DIdx(matches[i, 0])
    p3d_idx2 = im2.GetPoint3DIdx(matches[i, 1])
    if p3d_idx1 == -1 and p3d_idx2 == -1:
      new_matches = np.append(new_matches, matches[[i]], 0)

  num_new_matches = new_matches.shape[0]
  points3D = np.zeros((num_new_matches, 3))

  for i in range(num_new_matches):
    kp1 = im1.kps[new_matches[i, 0], :]
    kp2 = im2.kps[new_matches[i, 1], :]

    # H & Z Sec. 12.2
    A = np.array([
      kp1[0] * P1[2] - P1[0],
      kp1[1] * P1[2] - P1[1],
      kp2[0] * P2[2] - P2[0],
      kp2[1] * P2[2] - P2[1]
    ])

    _, _, vh = np.linalg.svd(A)
    homogeneous_point = vh[-1]
    points3D[i] = homogeneous_point[:-1] / homogeneous_point[-1]
     # We need to keep track of the correspondences between image points and 3D points
    im1_corrs = new_matches[:,0]
    im2_corrs = new_matches[:,1]

    # TODO
    # Filter points behind the cameras by transforming them into each camera space and checking the depth (Z)
    # Make sure to also remove the corresponding rows in `im1_corrs` and `im2_corrs`

    # transfom to camara space
    points3D_h = MakeHomogeneous(points3D, ax=1)
    camera1_points = P1 @ points3D_h.T
    camera2_points = P2 @ points3D_h.T
    
    valid_mask = (camera1_points[-1, :] > 0) & (camera2_points[-1, :] > 0)
    
    points3D = points3D[valid_mask]
    im1_corrs = im1_corrs[valid_mask]
    im2_corrs = im2_corrs[valid_mask]

    return points3D, im1_corrs, im2_corrs

def EstimateImagePose(points2D, points3D, K):
  # We use points in the normalized image plane.
  normalized_points2D = HNormalize(np.linalg.inv(K) @ MakeHomogeneous(points2D.T)).T

  constraint_matrix = BuildProjectionConstraintMatrix(normalized_points2D, points3D)

  # Solve for the nullspace
  _, _, vh = np.linalg.svd(constraint_matrix)
  P_vec = vh[-1, :]
  P = np.reshape(P_vec, (3, 4), order='C')

  # Make sure we have a proper rotation
  u, s, vh = np.linalg.svd(P[:, :3])
  R = u @ vh

  if np.linalg.det(R) < 0:
    R *= -1

  _, _, vh = np.linalg.svd(P)
  C = np.copy(vh[-1, :])

  t = -R @ (C[:3] / C[3])

  return R, t


def TriangulateImage(K, image_name, images, registered_images, matches):
  image = images[image_name]
  points3D = np.zeros((0, 3))
  corrs = {}

  for reg_image_name in registered_images:
    reg_image = images[reg_image_name]
    pair_matches = GetPairMatches(matches, image_name, reg_image_name)
    new_points3D, im1_corrs, im2_corrs = TriangulatePoints(K, image, reg_image, pair_matches)

    points3D = np.vstack((points3D, new_points3D))

    if image_name not in corrs:
      corrs[image_name] = []
    if reg_image_name not in corrs:
      corrs[reg_image_name] = []

    corrs[image_name].extend(im1_corrs)
    corrs[reg_image_name].extend(im2_corrs)

  return points3D, corrs
