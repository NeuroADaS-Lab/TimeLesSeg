import numpy as np
import nibabel as nib

from ..preprocessing import sample_foreground_locations_v2


def true_sampler(
    seg: np.ndarray,
    classes_or_regions: list[int | tuple[int]],
    seed: int,
    verbose: bool,
    min_num_samples: int,
    min_percent_coverage: float
):
    rndst = np.random.RandomState(seed)

    class_locs = {}

    # Normalize requested labels and compute the set of all labels we might need
    normalized = []
    requested_labels = set()
    for c in classes_or_regions:
        if isinstance(c, (tuple, list)):
            labs = tuple(int(x) for x in c)
            normalized.append(labs)
            requested_labels.update(labs)
        else:
            lab = int(c)
            normalized.append(lab)
            requested_labels.add(lab)

    # Create mask for all requested labels (this includes 0 if requested)
    requested_labels_arr = np.fromiter(requested_labels, dtype=np.int32)
    valid_mask = np.isin(seg, requested_labels_arr)

    coords = np.argwhere(valid_mask)
    seg_sel = seg[valid_mask]
    del valid_mask

    n = seg_sel.size
    if n == 0:
        for c in classes_or_regions:
            k = tuple(c) if isinstance(c, (tuple, list)) else int(c)
            class_locs[k] = []
        return class_locs

    # sort once, then compute label blocks
    order = np.argsort(seg_sel, kind="stable")
    lab_sorted = seg_sel[order]
    coords_sorted = coords[order]

    change = np.flatnonzero(lab_sorted[1:] != lab_sorted[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, n]
    labels_present = lab_sorted[starts]

    label_to_range = {int(l): (int(s), int(e)) for l, s, e in zip(labels_present, starts, ends)}
    present_labels = set(label_to_range.keys())

    for c in classes_or_regions:
        is_region = isinstance(c, (tuple, list))
        labs = tuple(int(x) for x in c) if is_region else (int(c),)
        k = labs if is_region else labs[0]

        # Skip if none of the labels are present
        if not any(lab in present_labels for lab in labs):
            class_locs[k] = []
            continue

        # Collect ranges for present labels in this class/region
        ranges = []
        counts = []
        for lab in labs:
            r = label_to_range.get(lab)
            if r is None:
                continue
            s, e = r
            cnt = e - s
            if cnt > 0:
                ranges.append((s, e))
                counts.append(cnt)

        if len(counts) == 0:
            class_locs[k] = []
            continue

        total = int(np.sum(counts))
        target_num_samples = min(min_num_samples, total)
        target_num_samples = max(target_num_samples, int(np.ceil(total * min_percent_coverage)))

        # Sample uniformly without replacement from the union of ranges, without building an n-sized mask
        # Draw target_num_samples unique offsets in [0, total)
        offsets = rndst.choice(total, target_num_samples, replace=False)

        # Map offsets -> (range index, in-range offset) using cumulative counts
        cum = np.cumsum(counts)
        which = np.searchsorted(cum, offsets, side="right")
        prev = np.concatenate(([0], cum[:-1]))
        in_range = offsets - prev[which]

        # Convert to indices in coords_sorted
        starts_for_pick = np.fromiter((ranges[i][0] for i in which), dtype=np.int64, count=which.size)
        picked_idx = starts_for_pick + in_range.astype(np.int64)

        selected = coords_sorted[picked_idx]
        class_locs[k] = selected

        if verbose:
            print(c, target_num_samples)

    return class_locs
    

def main(data_path: str, seg_path: str):
    # data = np.asanyarray(nib.load(data_path).dataobj)
    seg = np.asanyarray(nib.load(seg_path).dataobj).astype(np.int16)

    foreground_labels = list(np.unique(seg))[1:]
    seed = 1234
    verbose = False
    max_voxels_to_sample = 10_000
    min_coverage_ratio_per_class = 0.01

    seg = seg[None]
    
    res1 = true_sampler(seg, foreground_labels, seed, verbose, max_voxels_to_sample, min_coverage_ratio_per_class)
    res2 = sample_foreground_locations_v2(seg, foreground_labels, seed, verbose, max_voxels_to_sample, min_coverage_ratio_per_class)

    assert len(res1) == len(res2) and res1.keys() == res2.keys()
    for v1, v2 in zip(res1.values(), res2.values()):
        assert np.array_equal(v1, v2)
    
    res1 = true_sampler(np.zeros_like(seg), foreground_labels, seed, verbose, max_voxels_to_sample, min_coverage_ratio_per_class)
    res2 = sample_foreground_locations_v2(np.zeros_like(seg), foreground_labels, seed, verbose, max_voxels_to_sample, min_coverage_ratio_per_class)

    assert len(res1) == len(res2) and res1.keys() == res2.keys()
    for v1, v2 in zip(res1.values(), res2.values()):
        assert np.array_equal(v1, v2)
 
if __name__ == "__main__":
    data_path = '/Users/vicentcaselles/work/research/project_MARCOS/Multiple-Sclerosis-TIMILS/subj7/flair_bfc.nii.gz'
    seg_path = '/Users/vicentcaselles/work/research/project_MARCOS/Multiple-Sclerosis-TIMILS/subj7/flair_bfc_filled_NeuroMorph_Parcellation_cleaned.nii.gz'
    main(data_path, seg_path)