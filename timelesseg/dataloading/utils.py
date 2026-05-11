from __future__ import annotations
import re

from timelesseg.utils import (
    listdir,
    join,
    exists
)


def _sanity_check_identifiers(identifiers: list[str]):
    return len(identifiers) and all(len(identifiers[0]) == len(i) for i in identifiers[1:])

def _get_identifiers(folder: str, file_ending: str | tuple[str, ...]) -> set[str]:
    data_regex = re.compile(r'([IM][fb])?_(\d+)')
    files = filter(lambda x: x.endswith(file_ending), listdir(folder))
    return set(
        map(
            lambda x: x.groups()[-1],
            filter(None, map(
                data_regex.search,
                files
                )
            )
        )
    )

def get_identifiers(folder: str, file_ending: str | tuple[str, ...], sort: bool = True) -> list[str]:
    identifiers = list(_get_identifiers(folder, file_ending))
    if sort:
        identifiers.sort()
    return identifiers

def _get_image_fnames(folder, identifier, file_ending) -> tuple[str, str | None]:
    image_fnames = (
        'If_' + identifier + file_ending,
        'Mb_' + identifier + file_ending
    )
    # this sets Mb to None in case we don't have it. Preprocessing (should) handles addition of empty channel
    image_fnames = tuple(map(lambda x: join(folder, x) if exists(join(folder, x)) else None, image_fnames))
    return image_fnames

def _generate_iterable_with_fnames(folder: str, identifier: str, allow_no_seg: bool, file_ending: str) -> dict[str, str | tuple[str, str | None]]:
    """
    :param allow_no_seg: enforces non-available ground truths.
    """

    image_fnames = _get_image_fnames(folder, identifier, file_ending)

    seg_fname = join(folder, 'Mf_' + identifier + file_ending)
    if not exists(seg_fname):
        if not allow_no_seg:
            raise FileNotFoundError('seg "%s" not found' % seg_fname)

        seg_fname = None

    assert exists(image_fnames[0]), 'At least one input has to be given to the net... If: %s' % image_fnames[0]

    return {'images': image_fnames, 'seg': seg_fname, 'identifier': identifier}

def _generate_iterable_with_fnames_return_dict(folder: str, allow_no_seg: bool = True, file_ending: str = '.nii.gz') -> dict[str, dict[str, tuple[str | None] | str | None]]:
    return {ii: _generate_iterable_with_fnames(folder, ii, allow_no_seg, file_ending) for ii in get_identifiers(folder, file_ending)}

def generate_iterable_with_fnames(folder: str,
                                  allow_no_seg: bool,
                                  file_ending: str = '.nii.gz',
                                  return_dict: bool = True):
    """
    :param allow_no_seg:  it just means whether we allow seg to be None
    :return: dictionary with the following form: {'identifier': {'images': tuple(str, str | None), 'seg': str}}
    """
    iterable_as_dict = _generate_iterable_with_fnames_return_dict(folder, allow_no_seg, file_ending)
    if return_dict:
        return iterable_as_dict

    return list(iterable_as_dict.values())

if __name__ == "__main__":
    # x = get_identifiers('test-out', '.nii.gz')
    # assert (len(x) == 9000) and (x[0] == '0001'), x[0]

    # my_xx = generate_iterable_with_fnames('test-out', '.nii.gz')
    # assert len(my_xx.keys()) == 9000

    print(generate_iterable_with_fnames('validation_data/raw'))