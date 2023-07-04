import os
import os.path as osp
import sys
import numpy as np
import pandas as pd
import nilearn
from nilearn import plotting, image
from nilearn.maskers import NiftiLabelsMasker
from nilearn.regions import connected_label_regions
import time
from tqdm import tqdm
import torch

def get_subcortex_indices(subcortex_label_path = "/home/surprise/YAD_STAGIN/data/rois/Atlas_ROIs.2.txt"):
    subcortex_indices = []
    subcortex_names = []
    with open(subcortex_label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            idx = int(line.split('  ')[0].strip())
            name = line.split('  ')[1].strip()
            subcortex_indices.append(idx)
            subcortex_names.append(name)
            #print(idx, name)
    return subcortex_indices, subcortex_names



if __name__=='__main__':
        
    start = time.time()
    embarc_base = "/home/drbong_EMBARC/EMBARC"
    bids_dir = osp.join(embarc_base, "bids")

    subject_ids = [ file for file in os.listdir(bids_dir) if file.startswith("sub-") ]
    timeseries_dict = {}
    for subject_id in tqdm(subject_ids):
        print(subject_id)
        start = time.time()
        # set paths
        func_dir = osp.join(embarc_base, 'output', subject_id, 'ses-1', 'func')
        func_file = f"{subject_id}_ses-1_task-rest_run-1_space-MNI152NLin6Asym_desc-smoothAROMAnonaggr_bold.nii.gz"
        func_file_full = osp.join(func_dir, func_file)
        aseg_file = f'{subject_id}_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_res-1_desc-aseg_dseg.nii.gz' 
        aseg_file_full = osp.join(func_dir, aseg_file)
        confounder_file = f"{subject_id}_ses-1_task-rest_run-1_desc-confounds_timeseries.tsv"
        confounder_file_full = osp.join(func_dir, confounder_file)
        if osp.exists(confounder_file_full):
            confounds_all = pd.read_table(confounder_file_full)
            confounds = confounds_all.dropna(axis='columns')[['global_signal', 'csf', 'white_matter', 'csf_wm']]
        else: print("NO confouder files")
        cortical_label_path = "/home/surprise/YAD_STAGIN/data/rois/schaefer100-yeo17.txt"
        cortical_labels = pd.read_csv(cortical_label_path, sep="  ")[['KEY', ' NAME']]  # should be checked with nilearn atlas
        
        if osp.exists(func_file_full) and osp.exists(aseg_file_full):
            # timeseries for cortex
            schaefer100 = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=17, resolution_mm=1, data_dir=None, base_url=None, resume=True, verbose=1)
            #plotting.plot_roi(schaefer100['maps'], colorbar=True)
            cortex_names = schaefer100['labels']
            cortex_names = [ label.decode() for label in cortex_names]
            masker_cortex = NiftiLabelsMasker(labels_img=schaefer100['maps'], standardize=True)
            ts_schaefer100 = masker_cortex.fit_transform(func_file_full, confounds=confounds)

            # timeseries for subcortex
            # aseg_plot = plotting.plot_roi(aseg_file_full, cut_coords=(36, -27, -20), colorbar=True, cmap='Paired')
            subcortex_indices, subcortex_names = get_subcortex_indices()
            aseg_label_img = image.load_img(aseg_file_full)
            labels = aseg_label_img.get_fdata()
            labels_subcortex19 = labels.copy()
            labels_subcortex19[~np.isin(labels, subcortex_indices)]=0
            label_img_subcortex19 = image.new_img_like(aseg_label_img, labels_subcortex19)
            masker_subcortex19 = NiftiLabelsMasker(labels_img=label_img_subcortex19, labels=subcortex_names, standardize=True)
            ts_subcortex19 = masker_subcortex19.fit_transform(func_file_full, confounds=confounds)

            # concatenate
            timeseries = np.concatenate([ts_schaefer100, ts_subcortex19], axis=1).T   # node x timepoints
            labels = cortex_names + subcortex_names
            

            timeseries_dict.update({subject_id.split('-')[-1]:timeseries})
            #break
        else:
            if not osp.exists(func_file_full): print(f"No fMRI files: {func_file_full}")
            if not osp.exists(aseg_file_full): print(f"No Aseg files: {aseg_file_full}")
    
    print(f"Elapsed: {time.time()- start} sec.")
    torch.save(timeseries_dict, "EMBARC_schaefer100_sub19_ses1.pth")