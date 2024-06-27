import numpy as np
import nibabel as nib
from nilearn.datasets import fetch_atlas_destrieux_2009
from analyses.ridge_regression_decoding import REGIONS_LOW_LEVEL_VISUAL, REGIONS_HIGH_LEVEL_VISUAL, REGIONS_LANGUAGE
import pandas as pd


def run():
    long_names = {}
    ids = {}
    with open("destrieux.txt") as file:
        for line in file:
            line = line.rstrip()
            long_name = " ".join(", ".join(line.split(', ')[1:]).split(' ')[1:])
            # pattern = r'[0-9]'
            # long_name = re.sub(pattern, '', long_name)
            name = line.split(', ')[1].split(' ')[0]
            long_names[name] = long_name
            ids[name] = int(line.split(',')[0])

    destrieux_atlas = fetch_atlas_destrieux_2009()
    label_to_value_dict = {label[1]: int(label[0]) for label in destrieux_atlas['labels']}
    atlas_map = nib.load(destrieux_atlas.maps).get_fdata()

    roi_descs = {}
    roi_names = ["visual_low_level", "visual_high_level", "language_network"]
    for roi_name, region in zip(roi_names, [REGIONS_LOW_LEVEL_VISUAL, REGIONS_HIGH_LEVEL_VISUAL, REGIONS_LANGUAGE]):
        region_names = [label for label in region]
        values = [label_to_value_dict[label] for label in region_names]
        roi_mask = np.isin(atlas_map, values)
        roi_mask_size = roi_mask.sum()
        print(roi_name)
        print("ROI size: {} voxels".format(roi_mask_size))
        print("Parts:")
        regions = []
        for name, id in zip(region_names, values):
            region_mask = np.isin(atlas_map, id)
            region_mask_size = region_mask.sum()
            regions.append({"id": ids[name[2:]], "label": name, "names": long_names[name[2:]]})
            print(f"\tID: {ids[name[2:]]:02d} num voxels: {region_mask_size:04d} names: {name}:\t({name[0]} {long_names[name[2:]]})")
        print()
        roi_descs[roi_name] = pd.DataFrame.from_records(regions, index="id")

    for roi_name in roi_names:
        print(roi_name)
        print(roi_descs[roi_name].to_latex(escape=True))


if __name__ == "__main__":
    run()
