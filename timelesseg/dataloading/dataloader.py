from __future__ import annotations
from abc import ABCMeta, abstractmethod
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from timelesseg.preprocessing.cropping import crop_and_pad_nd

from .dataset import Dataset


class SlimDataLoaderBase:
    def __init__(
        self,
        data,
        batch_size,
        number_of_threads_in_multithreaded = None
    ):
        """
        Slim version of DataLoaderBase (which is now deprecated). Only provides very simple functionality.

        You must derive from this class to implement your own DataLoader. You must overrive self.generate_train_batch()

        If you use our MultiThreadedAugmenter you will need to also set and use number_of_threads_in_multithreaded. See
        multithreaded_dataloading in examples!

        :param data: will be stored in self._data. You can use it to generate your batches in self.generate_train_batch()
        :param batch_size: will be stored in self.batch_size for use in self.generate_train_batch()
        :param number_of_threads_in_multithreaded: will be stored in self.number_of_threads_in_multithreaded.
        None per default. If you wish to iterate over all your training data only once per epoch, you must coordinate
        your Dataloaders and you will need this information
        """
        __metaclass__ = ABCMeta
        self.number_of_threads_in_multithreaded = number_of_threads_in_multithreaded
        self._data = data
        self.batch_size = batch_size
        self.thread_id = 0

    def set_thread_id(self, thread_id):
        self.thread_id = thread_id

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_train_batch()

    @abstractmethod
    def generate_train_batch(self):
        '''override this
        Generate your batch from self._data .Make sure you generate the correct batch size (self.BATCH_SIZE)
        '''
        pass


class _DataLoader(SlimDataLoaderBase):
    def __init__(
        self,
        data,
        batch_size,
        num_threads_in_multithreaded=1,
        seed_for_shuffle=None,
        return_incomplete=False,
        shuffle=True,
        infinite=False,
        sampling_probabilities=None
    ):
        """
        :param data: will be stored in self._data for use in generate_train_batch
        :param batch_size: will be used by get_indices to return the correct number of indices
        :param num_threads_in_multithreaded: num_threads_in_multithreaded necessary for synchronization of dataloaders
        when using multithreaded augmenter
        :param seed_for_shuffle: for reproducibility
        :param return_incomplete: whether or not to return batches that are incomplete. Only applies is infinite=False.
        If your data has len of 34 and your batch size is 32 then there return_incomplete=False will make this loader
        return only one batch of shape 32 (omitting 2 of your training examples). If return_incomplete=True a second
        batch with batch size 2 will be returned.
        :param shuffle: if True, the order of the indices will be shuffled between epochs. Only applies if infinite=False
        :param infinite: if True, each batch contains randomly (uniformly) sampled indices. An unlimited number of
        batches is returned. If False, DataLoader will iterate over the data only once
        :param sampling_probabilities: only applies if infinite=True. If sampling_probabilities is not None, the
        probabilities will be used by np.random.choice to sample the indexes for each batch. Important:
        sampling_probabilities must have as many entries as there are samples in your dataset AND
        sampling_probabilities needs to sum to 1
        """
        super(_DataLoader, self).__init__(data, batch_size, num_threads_in_multithreaded)
        self.infinite = infinite
        self.shuffle = shuffle
        self.return_incomplete = return_incomplete
        self.seed_for_shuffle = seed_for_shuffle
        self.rs = np.random.RandomState(self.seed_for_shuffle)
        self.current_position = None
        self.was_initialized = False
        self.last_reached = False
        self.sampling_probabilities = sampling_probabilities

        # when you derive, make sure to set this! We can't set it here because we don't know what data will be like
        self.indices = None

    def reset(self):
        assert self.indices is not None

        self.current_position = self.thread_id * self.batch_size

        self.was_initialized = True

        # no need to shuffle if we are returning infinite random samples
        if not self.infinite and self.shuffle:
            self.rs.shuffle(self.indices)

        self.last_reached = False

    def get_indices(self):
        # if self.infinite, this is easy
        if self.infinite:
            return np.random.choice(self.indices, self.batch_size, replace=True, p=self.sampling_probabilities)

        if self.last_reached:
            self.reset()
            raise StopIteration

        if not self.was_initialized:
            self.reset()

        indices = []

        for b in range(self.batch_size):
            if self.current_position < len(self.indices):
                indices.append(self.indices[self.current_position])

                self.current_position += 1
            else:
                self.last_reached = True
                break

        if len(indices) > 0 and ((not self.last_reached) or self.return_incomplete):
            self.current_position += (self.number_of_threads_in_multithreaded - 1) * self.batch_size
            return indices

        self.reset()
        raise StopIteration

    @abstractmethod
    def generate_train_batch(self):
        '''
        make use of self.get_indices() to know what indices to work on!
        :return:
        '''
        pass


class DataLoader(_DataLoader):
    _data: Dataset

    def __init__(self,
                 data: Dataset,
                 batch_size: int,
                 patch_size: list[int] | tuple[int] |np.ndarray,
                 final_patch_size: list[int] | tuple[int, ...] | np.ndarray,
                 all_labels: tuple[int],
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: list[int] | tuple[int] | np.ndarray = None,
                 pad_sides: list[int] | tuple[int] = None,
                 probabilistic_oversampling: bool = False,
                 transforms = None):
        """
        :param patch_size: The patch size we want taking into account transforms (rotation, etc...)
        :param final_patch_size: Actual patch size that the network gets!!
        """
        super().__init__(data, batch_size, 1, None, True,
                         False, True, sampling_probabilities)

        # this is used by DataLoader for sampling train cases!
        self.indices = data.identifiers

        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.all_labels = tuple([-1] + all_labels)

        # patch size (required due to rotation etc) is usually greater than final_patch_size.
        # So at least we need to pad to cover their difference!
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            for d in range(len(self.need_to_pad)):
                self.need_to_pad[d] += pad_sides[d]
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.sampling_probabilities = sampling_probabilities
        self.get_do_oversample = self._oversample_last_XX_percent if not probabilistic_oversampling \
            else self._probabilistic_oversampling
        self.transforms = transforms

    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        """
        determines whether sample sample_idx in a minibatch needs to be guaranteed foreground
        """
        return not (sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent)))

    def _probabilistic_oversampling(self, sample_idx: int) -> bool:
        # print('YEAH BOIIIIII')
        return np.random.uniform() < self.oversample_foreground_percent

    def determine_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """
        Gives you the data shape that the network will get (i.e., batch, channels, patchsize[X, Y, Z])
        """
        # load one case
        data, seg, properties = self._data.load_case(self._data.identifiers[0])
        num_color_channels = data.shape[0]

        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        channels_seg = seg.shape[0]
        seg_shape = (self.batch_size, channels_seg, *self.patch_size)
        return data_shape, seg_shape


    def get_bbox(
        self,
        data_shape: np.ndarray,
        force_fg: bool,
        class_locations: dict | None,
        overwrite_class: int | tuple[int, ...] = None,
        verbose: bool = False
    ):

        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            # print('I want a random location')
        else:
            assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
            if overwrite_class is not None:
                assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not ' \
                                                                    'have class_locations (missing key)'
            # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
            # class_locations keys can also be tuple.
            # Remember, class locations only has foreground classes! This removes any background that might be in
            # self.all_labels
            eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]

            # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
            # strange formulation needed to circumvent
            # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
            tmp = [i == self.all_labels if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
            if any(tmp):
                if len(eligible_classes_or_regions) > 1:
                    eligible_classes_or_regions.pop(np.where(tmp)[0][0])

            if len(eligible_classes_or_regions) == 0:
                # this only happens if some image does not contain foreground voxels at all
                selected_class = None
                if verbose:
                    print('case does not contain any foreground classes')
            else:
                # I hate myself. Future me aint gonna be happy to read this
                # 2022_11_25: had to read it today. Wasn't too bad
                selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                    (overwrite_class is None or (overwrite_class not in eligible_classes_or_regions)) else overwrite_class
            # print(f'I want to have foreground, selected class: {selected_class}')

            if selected_class is not None:
                voxels_of_that_class = class_locations[selected_class]
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                # i + 1 because we have first dimension 0!
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                # If the image does not contain any foreground classes, we fall back to random cropping
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs


    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]

            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            # use ACVL utils for that. Cleaner.
            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)

            seg_all[j] = seg_cropped

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images

        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}



class DataLoaderDownsamplesChannel1(DataLoader):
    def __init__(self,
                 data: Dataset,
                 batch_size: int,
                 patch_size: list[int] | tuple[int] |np.ndarray,
                 final_patch_size: list[int] | tuple[int, ...] | np.ndarray,
                 all_labels: tuple[int],
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: list[int] | tuple[int] | np.ndarray = None,
                 pad_sides: list[int] | tuple[int] = None,
                 probabilistic_oversampling: bool = False,
                 transforms = None,
                 deep_supervision_scales: bool = None):
        """
        :param patch_size: The patch size we want taking into account transforms (rotation, etc...)
        :param final_patch_size: Actual patch size that the network gets!!
        """
        super().__init__(data, batch_size, patch_size, final_patch_size,
                         all_labels, oversample_foreground_percent, sampling_probabilities,
                         pad_sides, probabilistic_oversampling, transforms)

        # transforms __call__ gets a data dict
        self._last_transform = lambda **data_dict: data_dict
        if deep_supervision_scales is not None:
            from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
            self._last_transform = DownsampleSegForDSTransform(deep_supervision_scales)

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]

            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            # use ACVL utils for that. Cleaner.
            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)

            seg_all[j] = seg_cropped

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    baselines = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])

                        # get first channel of data, [[1]] copies it, set it to int16
                        baseline_tmp = tmp['image'][[1]].to(seg_all.dtype, copy=True)
                        baselines.append(self._last_transform(**{'segmentation': baseline_tmp})['segmentation'])

                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    if isinstance(baselines[0], list):
                        baseline_all = [torch.stack([bb[i] for bb in baselines]).to(torch.int16) for i in range(len(baselines[0]))]
                    else:
                        baseline_all = torch.stack(baselines).to(torch.int16)
                    del segs, images, baselines
            return {'data': data_all, 'target': seg_all, 'baseline': baseline_all, 'keys': selected_keys}

        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}
