import os
import numpy as np
import pandas as pd
import skimage as ski
from skimage import measure
import nibabel as nib
import glob
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_patient_segmentation(patient_id, masks_path):
    """Load and combine all segmentation files for a patient."""
    seg_files = sorted(glob.glob(os.path.join(masks_path, f"{patient_id}_SEG.nii-*.nii.gz")))

    if not seg_files:
        return None, None

    print(f"    Found {len(seg_files)} segment files, combining...")

    combined_mask = None
    mask_affine = None

    for seg_file in seg_files:
        seg_nii = nib.load(seg_file)
        seg_data = seg_nii.get_fdata()

        if combined_mask is None:
            combined_mask = seg_data > 0
            mask_affine = seg_nii.affine
        else:
            combined_mask = combined_mask | (seg_data > 0)

    return combined_mask, mask_affine

def resample_mask_to_ct(mask_data, mask_affine, ct_nii):
    """Resample mask to match CT image dimensions and orientation."""
    from scipy.ndimage import zoom

    ct_data = ct_nii.get_fdata()
    ct_shape = ct_data.shape
    mask_shape = mask_data.shape

    if mask_shape == ct_shape:
        return mask_data

    zoom_factors = [ct_shape[i] / mask_shape[i] for i in range(3)]
    resampled_mask = zoom(mask_data.astype(np.float32), zoom_factors, order=0)
    resampled_mask = resampled_mask > 0.5

    return resampled_mask

def tumour_features(tumour_array, voxel_size):
    """Extract shape-based features."""
    if np.sum(tumour_array) == 0:
        return {
            "volume": 0, "surface_area": 0, "max_diameter": 0,
            "sphericity": 0, "compactness": 0, "elongation": 0
        }

    verts, faces, _, _ = ski.measure.marching_cubes(tumour_array, 0.5, spacing=voxel_size)
    area = ski.measure.mesh_surface_area(verts, faces)
    volume = np.sum(tumour_array) * np.prod(voxel_size)

    # Maximum diameter (bounding box diagonal)
    max_diameter = np.sqrt(np.sum(np.ptp(verts, axis=0)**2))

    # Sphericity: (36 * pi * V^2)^(1/3) / surface_area
    # Perfect sphere = 1
    sphericity = ((36 * np.pi * volume**2) ** (1/3)) / area if area > 0 else 0

    # Compactness: volume / (sqrt(pi) * (max_diameter/2)^3)
    # Lower = more elongated, Higher = more compact
    compactness = volume / ((4/3) * np.pi * (max_diameter/2)**3) if max_diameter > 0 else 0

    # Elongation: ratio of longest to shortest axis
    ranges = np.ptp(verts, axis=0)
    elongation = np.max(ranges) / (np.min(ranges) + 1e-10) if np.min(ranges) > 0 else 1

    return {
        "volume": float(volume),
        "surface_area": float(area),
        "max_diameter": float(max_diameter),
        "sphericity": float(sphericity),
        "compactness": float(compactness),
        "elongation": float(elongation)
    }

def intensity_features(img, mask):
    """Extract intensity-based features from CT image within tumor."""
    if np.sum(mask) < 10:
        return {
            "mean_intensity": 0, "std_intensity": 0,
            "skewness": 0, "kurtosis": 0, "entropy": 0, "energy": 0
        }

    masked_img = img[mask]

    if len(masked_img) == 0:
        return {
            "mean_intensity": 0, "std_intensity": 0,
            "skewness": 0, "kurtosis": 0, "entropy": 0, "energy": 0
        }

    # Basic statistics
    mean_val = np.mean(masked_img)
    std_val = np.std(masked_img)

    # Skewness and Kurtosis (measure of distribution shape)
    if std_val > 0:
        skewness = np.mean(((masked_img - mean_val) / std_val) ** 3)
        kurtosis = np.mean(((masked_img - mean_val) / std_val) ** 4) - 3
    else:
        skewness = 0
        kurtosis = 0

    # Entropy (measure of disorder/complexity)
    # Discretize into 32 bins for entropy calculation
    hist, _ = np.histogram(masked_img, bins=32, density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist + 1e-10))

    # Energy (uniformity)
    energy = np.sum(hist ** 2)

    return {
        "mean_intensity": float(mean_val),
        "std_intensity": float(std_val),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "entropy": float(entropy),
        "energy": float(energy)
    }

def gray_level_cooccurrence_features(img, mask):
    """Extract texture features (no correlation)."""
    if np.sum(mask) < 100:
        return {"contrast": 0, "dissimilarity": 0, "homogeneity": 0}

    masked_img = img[mask]
    if len(masked_img) < 100:
        return {"contrast": 0, "dissimilarity": 0, "homogeneity": 0}

    # Normalize to 0-255 range
    img_min = np.amin(masked_img)
    img_max = np.amax(masked_img)

    if img_max <= img_min:
        return {"contrast": 0, "dissimilarity": 0, "homogeneity": 0}

    img_norm = (img - img_min) / (img_max - img_min) * 255
    img_quantized = np.floor(img_norm).astype(int)

    # Use 32 bins for GLCM
    n_levels = 32
    img_discrete = np.floor(img_quantized / (256 / n_levels)).astype(int)
    img_discrete = np.clip(img_discrete, 0, n_levels - 1)

    glcm = _calculate_glcm2(img_discrete, mask, n_levels)

    if np.sum(glcm) == 0:
        return {"contrast": 0, "dissimilarity": 0, "homogeneity": 0}

    glcm = glcm / np.sum(glcm, axis=(0, 1))

    i_indices = np.arange(n_levels)[:, np.newaxis, np.newaxis]
    j_indices = np.arange(n_levels)[np.newaxis, :, np.newaxis]

    contrast = np.mean(np.sum(((i_indices - j_indices) ** 2) * glcm, axis=(0, 1)))
    dissimilarity = np.mean(np.sum(np.abs(i_indices - j_indices) * glcm, axis=(0, 1)))
    homogeneity = np.mean(np.sum(glcm / (1 + np.abs(i_indices - j_indices)), axis=(0, 1)))

    return {
        "contrast": float(contrast),
        "dissimilarity": float(dissimilarity),
        "homogeneity": float(homogeneity)
    }

def _calculate_glcm2(img, mask, nbins):
    """Calculate GLCM for 13 different 3D directions."""
    out = np.zeros((nbins, nbins, 13))
    offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (-1, 1, 0),
               (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1),
               (-1, 1, 1), (1, -1, 1), (1, 1, -1)]

    matrix = np.array(img)
    matrix[mask <= 0] = nbins
    s = matrix.shape
    bins = np.arange(0, nbins + 1)

    for i, offset in enumerate(offsets):
        matrix1 = np.ravel(matrix[
            max(offset[0], 0):s[0] + min(offset[0], 0),
            max(offset[1], 0):s[1] + min(offset[1], 0),
            max(offset[2], 0):s[2] + min(offset[2], 0)
        ])
        matrix2 = np.ravel(matrix[
            max(-offset[0], 0):s[0] + min(-offset[0], 0),
            max(-offset[1], 0):s[1] + min(-offset[1], 0),
            max(-offset[2], 0):s[2] + min(-offset[2], 0)
        ])
        out[:, :, i] = np.histogram2d(matrix1, matrix2, bins=bins)[0]
    return out

def main():
    nifti_images_path = '/user/work/va24397/summative_AIHDS/nifti_images'
    nifti_masks_path = '/user/work/va24397/summative_AIHDS/nifti_masks'
    output_dir = '/user/work/va24397/summative_AIHDS'

    all_patient_ids = []
    for f in os.listdir(nifti_images_path):
        if f.endswith('_CT.nii.gz'):
            all_patient_ids.append(f.replace('_CT.nii.gz', ''))

    all_patient_ids = sorted(all_patient_ids)
    print(f"Found {len(all_patient_ids)} total patients with CT images")

    # ========== TEST MODE: Process only first 5 patients ==========
    TEST_MODE = False
    TEST_PATIENTS = 5

    if TEST_MODE:
        patient_ids = all_patient_ids[:TEST_PATIENTS]
        print(f"\nTEST MODE: Processing first {len(patient_ids)} patients only")
        print(f"Patients to test: {patient_ids}")
    else:
        patient_ids = all_patient_ids
        print(f"\nProcessing all {len(patient_ids)} patients")
    # ===============================================================

    print("="*50)

    valid_patients = []
    features_shape = []
    features_intensity = []
    features_texture = []

    for i, patient_id in enumerate(patient_ids):
        print(f"\n{'='*50}")
        print(f"Processing {patient_id} ({i+1}/{len(patient_ids)})...")

        ct_path = os.path.join(nifti_images_path, f"{patient_id}_CT.nii.gz")
        if not os.path.exists(ct_path):
            print(f"  CT file not found - skipping")
            continue

        ct_nii = nib.load(ct_path)
        ct_data = ct_nii.get_fdata()
        voxel_size = ct_nii.header.get_zooms()

        combined_mask, mask_affine = load_patient_segmentation(patient_id, nifti_masks_path)

        if combined_mask is None:
            print(f"  No segmentation files found - skipping")
            continue

        combined_mask = resample_mask_to_ct(combined_mask, mask_affine, ct_nii)

        tumor_voxels = np.sum(combined_mask)
        if tumor_voxels == 0:
            print(f"  Segmentation is empty - skipping")
            continue

        print(f"  Tumor size: {tumor_voxels} voxels")

        shape_feats = tumour_features(combined_mask, voxel_size)
        intensity_feats = intensity_features(ct_data, combined_mask)
        texture_feats = gray_level_cooccurrence_features(ct_data, combined_mask)

        features_shape.append(list(shape_feats.values()))
        features_intensity.append(list(intensity_feats.values()))
        features_texture.append(list(texture_feats.values()))
        valid_patients.append(patient_id)

        print(f"  Volume: {shape_feats['volume']:.1f} mm^3")
        print(f"  Surface area: {shape_feats['surface_area']:.1f} mm^2")
        print(f"  Sphericity: {shape_feats['sphericity']:.3f}")
        print(f"  Mean intensity: {intensity_feats['mean_intensity']:.1f} HU")
        print(f"  Contrast: {texture_feats['contrast']:.3f}")
        print(f"  Homogeneity: {texture_feats['homogeneity']:.3f}")

    print(f"\n{'='*50}")
    print(f"PROCESSING COMPLETE")
    print(f"Successfully processed {len(valid_patients)}/{len(patient_ids)} patients")

    if len(valid_patients) > 0:
        shape_names = ["volume", "surface_area", "max_diameter", "sphericity", "compactness", "elongation"]
        intensity_names = ["mean_intensity", "std_intensity", "skewness", "kurtosis", "entropy", "energy"]
        texture_names = ["contrast", "dissimilarity", "homogeneity"]

        # Add test_ prefix to filenames in test mode
        if TEST_MODE:
            shape_filename = os.path.join(output_dir, "test_features_shape.csv")
            intensity_filename = os.path.join(output_dir, "test_features_intensity.csv")
            texture_filename = os.path.join(output_dir, "test_features_texture.csv")
        else:
            shape_filename = os.path.join(output_dir, "features_shape.csv")
            intensity_filename = os.path.join(output_dir, "features_intensity.csv")
            texture_filename = os.path.join(output_dir, "features_texture.csv")

        df_shape = pd.DataFrame(np.array(features_shape), columns=shape_names)
        df_shape['patient_id'] = valid_patients
        df_shape.set_index('patient_id', inplace=True)
        df_shape.to_csv(shape_filename)
        print(f"\nSaved: {shape_filename}")

        df_intensity = pd.DataFrame(np.array(features_intensity), columns=intensity_names)
        df_intensity['patient_id'] = valid_patients
        df_intensity.set_index('patient_id', inplace=True)
        df_intensity.to_csv(intensity_filename)
        print(f"Saved: {intensity_filename}")

        df_texture = pd.DataFrame(np.array(features_texture), columns=texture_names)
        df_texture['patient_id'] = valid_patients
        df_texture.set_index('patient_id', inplace=True)
        df_texture.to_csv(texture_filename)
        print(f"Saved: {texture_filename}")

        print(f"\nTotal features extracted: {len(shape_names) + len(intensity_names) + len(texture_names)}")
        print(f"  - Shape features: {len(shape_names)}")
        print(f"  - Intensity features: {len(intensity_names)}")
        print(f"  - Texture features: {len(texture_names)}")

        if TEST_MODE:
            print(f"\nTEST COMPLETE - Check the test CSV files above.")
            print(f"If results look good, change TEST_MODE to False and rerun for all patients.")
    else:
        print("\nNo patients were successfully processed!")
        print("Please check your file paths and segmentation files.")

if __name__ == '__main__':
    main()
