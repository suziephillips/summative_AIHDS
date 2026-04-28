import pydicom
if not hasattr(pydicom, 'read_file'):
    pydicom.read_file = pydicom.dcmread

import os
import dicom2nifti
import subprocess
import numpy as np
import nibabel as nib

def find_ct_folder(session_dir):
    """Find the CT folder containing multiple DICOM files."""
    for item in os.listdir(session_dir):
        item_path = os.path.join(session_dir, item)
        if os.path.isdir(item_path):
            for file in os.listdir(item_path)[:5]:
                file_path = os.path.join(item_path, file)
                if os.path.isfile(file_path):
                    try:
                        ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                        if hasattr(ds, 'Modality') and ds.Modality == 'CT':
                            print(f"Found CT folder: {item}")
                            return item_path
                    except:
                        continue
    return None

def find_seg_file(session_dir):
    """Find the SEG folder and return the path to its .dcm file."""
    for item in os.listdir(session_dir):
        item_path = os.path.join(session_dir, item)
        if os.path.isdir(item_path):
            for file in os.listdir(item_path):
                file_path = os.path.join(item_path, file)
                if os.path.isfile(file_path) and (file.endswith('.dcm') or file == '1-1.dcm'):
                    try:
                        ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                        if hasattr(ds, 'Modality') and ds.Modality == 'SEG':
                            print(f"Found SEG file: {item}/{file}")
                            return file_path
                    except:
                        continue
    return None

def convert_ct_to_nifti(ct_dicom_dir, output_nifti_file):
    """Convert a DICOM CT series to a NIfTI file."""
    print(f"Converting CT to NIfTI")
    try:
        # Try the standard conversion first
        dicom2nifti.dicom_series_to_nifti(ct_dicom_dir, output_nifti_file, reorient_nifti=True)
    except Exception as e:
        if "out of bounds for uint16" in str(e):
            print(f"Handling uint16 overflow by forcing float64 conversion...")
            # Alternative: Use nibabel directly with custom scaling
            convert_ct_direct(ct_dicom_dir, output_nifti_file)
        else:
            raise

def convert_ct_direct(ct_dicom_dir, output_nifti_file):
    """Direct conversion using nibabel for problematic CT series."""
    # Read all DICOM files in the folder
    dcm_files = sorted([f for f in os.listdir(ct_dicom_dir) if f.endswith('.dcm')])
    if not dcm_files:
        dcm_files = sorted([f for f in os.listdir(ct_dicom_dir) if os.path.isfile(os.path.join(ct_dicom_dir, f))])

    slices = []
    for f in dcm_files:
        ds = pydicom.dcmread(os.path.join(ct_dicom_dir, f))
        # Apply RescaleSlope and RescaleIntercept to get HU
        hu = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        slices.append(hu)

    # Stack into 3D volume
    volume = np.stack(slices, axis=0)

    # Get affine transformation from DICOM
    # This is simplified - you may need to compute proper affine
    affine = np.eye(4)

    # Save as NIfTI with float64 dtype to handle negative values
    nifti_img = nib.Nifti1Image(volume.astype(np.float64), affine)
    nib.save(nifti_img, output_nifti_file)

def convert_seg_to_nifti(seg_dicom_file, output_nifti_file):
    """Convert a DICOM SEG file to a NIfTI file using corrected dcmqi syntax."""
    print(f"Converting SEG to NIfTI...")

    # Get the base filename without extension for the stem
    stem = os.path.splitext(os.path.basename(output_nifti_file))[0]
    output_dir = os.path.dirname(output_nifti_file)

    cmd = [
        "segimage2itkimage",
        "--inputDICOM", seg_dicom_file,
        "--outputDirectory", output_dir,
        "-p", stem, 
        "-t", "nii"
    ]

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"dcmqi stderr: {result.stderr}")
        raise Exception(f"dcmqi failed with return code {result.returncode}")

    # dcmqi creates a file named like 'stem.nii' - check for it
    expected_output = os.path.join(output_dir, f"{stem}.nii")
    if os.path.exists(expected_output):
        # Convert to .nii.gz if needed, or just rename
        if expected_output != output_nifti_file:
            # If output expects .nii.gz, compress it
            if output_nifti_file.endswith('.gz'):
                import gzip
                import shutil
                with open(expected_output, 'rb') as f_in:
                    with gzip.open(output_nifti_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(expected_output)
                print(f"Compressed to {os.path.basename(output_nifti_file)}")
            else:
                os.rename(expected_output, output_nifti_file)
                print(f"Renamed to {os.path.basename(output_nifti_file)}")

def main():
    base_path = '/user/work/ms13525/mshds-ml-data-2026/dataset1'
    nifti_images_path = '/user/work/va24397/summative_AIHDS/nifti_images'
    nifti_masks_path = '/user/work/va24397/summative_AIHDS/nifti_masks'

    os.makedirs(nifti_images_path, exist_ok=True)
    os.makedirs(nifti_masks_path, exist_ok=True)

    # Get list of patient folders
    patient_folders = [f for f in sorted(os.listdir(base_path)) 
                      if os.path.isdir(os.path.join(base_path, f)) and f.startswith('LUNG1-')]

    print(f"Found {len(patient_folders)} patient folders")
    # ========== RESUME FROM SPECIFIC PATIENT remove afterwards ==========
    START_FROM = 'LUNG1-375'  # CHANGE THIS to the patient you want to start from

    # Find the index of the patient to start from
    try:
        start_index = patient_folders.index(START_FROM)
        patient_folders = patient_folders[start_index:]
        print(f"Resuming from {START_FROM} (patient {start_index+1} of {start_index + len(patient_folders)})")
        print(f"Patients to process in this run: {len(patient_folders)}")
    except ValueError:
        print(f"Warning: {START_FROM} not found! Starting from beginning.")
#end of lung300 addtion

    print(f"Output directories:")
    print(f"CT NIfTI: {nifti_images_path}")
    print(f"SEG NIfTI: {nifti_masks_path}")

   #TEST MODE: Process only first 2 patients
#    patient_folders = patient_folders[:2]
 #   print(f"\nTESTING MODE: Processing first {len(patient_folders)} patients only")

    processed = 0
    skipped = []

    for patient_id in patient_folders:
        patient_dir = os.path.join(base_path, patient_id)
        print(f"\n{'='*60}")
        print(f"Processing {patient_id}...")

        session_folders = [f for f in os.listdir(patient_dir) 
                          if os.path.isdir(os.path.join(patient_dir, f))]

        if not session_folders:
            print(f"No session folder found")
            skipped.append(f"{patient_id}: no session folder")
            continue

        session_dir = os.path.join(patient_dir, session_folders[0])
        print(f"Session: {session_folders[0]}")

        # Find and convert CT folder
        ct_folder = find_ct_folder(session_dir)
        if ct_folder is None:
            print(f"CT folder not found")
            skipped.append(f"{patient_id}: CT folder not found")
            continue

        ct_output = os.path.join(nifti_images_path, f"{patient_id}_CT.nii.gz")
        try:
            convert_ct_to_nifti(ct_folder, ct_output)
            print(f"CT saved: {os.path.basename(ct_output)}")
        except Exception as e:
            print(f"CT conversion failed: {e}")
            skipped.append(f"{patient_id}: CT conversion error")
            continue

        # Find and convert SEG file
        seg_file = find_seg_file(session_dir)
        if seg_file is None:
            print(f"SEG file not found")
            skipped.append(f"{patient_id}: SEG file not found")
            continue

        seg_output = os.path.join(nifti_masks_path, f"{patient_id}_SEG.nii.gz")
        try:
            convert_seg_to_nifti(seg_file, seg_output)
            print(f"SEG saved: {os.path.basename(seg_output)}")
        except Exception as e:
            print(f"SEG conversion failed: {e}")
            skipped.append(f"{patient_id}: SEG conversion error")
            continue

        processed += 1
        print(f"Successfully processed {patient_id}")

    print("\n" + "="*60)
    print(f"PROCESSING COMPLETE")
    print(f"Successfully processed: {processed}/{len(patient_folders)} patients")
    print(f"Skipped: {len(skipped)} patients")

if __name__ == "__main__":
    main()
