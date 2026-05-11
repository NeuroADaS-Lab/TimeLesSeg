from collections.abc import Iterable

def _processs_case(image_fpaths: Iterable[str]):
    for f in image_fpaths:
        print(f)

def process_case(image_fpaths: Iterable[str]):
    return _processs_case(filter(None, image_fpaths))

def main():
    dicts = {'1': {'images': ('somepath1.nii.gz', 'somepath2.nii.gz'), 'seg': 'path_to_seg.nii.gz'},
             '2': {'images': ('somepath3.nii.gz', None), 'seg': 'path_to_seg2.nii.gz'}}
    for d in dicts:
        process_case(dicts[d]['images'])

    print(dicts)

if __name__ == "__main__":
    main()