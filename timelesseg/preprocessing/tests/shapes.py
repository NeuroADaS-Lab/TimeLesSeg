
from timelesseg.experiment_planning.experiment_planning import compute_new_shape
from timelesseg.dataloading import Dataset
from timelesseg.preprocessing import get_preprocessing_config_from_dataset_fingerprint

# we want to test that:
# shapes recently computed from dataset_fingerprint and
# saved data shapes and
# shapes found in properties file
# all match!

def main(folder):
    d = Dataset(folder)
    config = get_preprocessing_config_from_dataset_fingerprint(folder + '/dataset_fingerprint.json')
    new_shapes = [compute_new_shape(i, j, config.target_spacing) for i, j in zip(config.dataset_fingerprint['shapes_after_cropping'], config.dataset_fingerprint['spacings'])]
    for i, ident in enumerate(d.identifiers):
        data, seg, properties = d.load_case(ident)
        this_shape = new_shapes[i]
        assert (
            (this_shape == data.shape[1:]).all() and
            (this_shape == properties['shape_after_resampling']).all() and
            (this_shape == seg.shape[1:]).all()
        )

if __name__ == "__main__":
    main('test-out_pp')