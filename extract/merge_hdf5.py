import argparse
import os
from glob import glob
import h5py
import os.path as osp

def combine(feat_dir, desc_name, file_type):
    file_type = '_' + file_type if file_type is not None else ''
    file_name = f'{desc_name}{file_type}_local.hdf5'

    if os.path.exists(osp.join(feat_dir, file_name)):
        hdf5_file = h5py.File(os.path.join(feat_dir, file_name), 'r')
        print(f"Loaded {osp.join(feat_dir, file_name)}")
    else:
        splits = sorted(glob(osp.join(feat_dir, f'{desc_name}_xa?_local.hdf5')))
        if len(splits):
            images_per_hdf5 = []
            for chunk_file in splits:
                with h5py.File(chunk_file, 'r') as f:
                    images_per_hdf5.append(f['features'].shape[0])
                    shape = f['features'].shape[1:]
                    dtype = f['features'].dtype

            hdf5_file = h5py.File(os.path.join(feat_dir, file_name), 'w')
            hdf5_dataset = h5py.VirtualLayout(shape=(sum(images_per_hdf5), *shape), dtype=dtype)

            total = 0
            for i, (chunk_file, num_images) in enumerate(zip(splits, images_per_hdf5)):
                vsource = h5py.VirtualSource(
                    chunk_file, "features", shape=(num_images, *shape), dtype=dtype
                )
                hdf5_dataset[total : total + num_images] = vsource
                total += num_images

            hdf5_file.create_virtual_dataset('features', hdf5_dataset)
        else:
            raise "No splits to combine."

    dtype, shape = hdf5_file['features'].dtype, hdf5_file['features'].shape
    hdf5_file.close()
    return dtype, shape


def main():
    parser = argparse.ArgumentParser(description='Merge hdf5 files.')
    parser.add_argument('--dataset', help='Dataset name to load embeddings of.')
    parser.add_argument('--desc_name', default='dinov2', help='Embeddings to load based on name.')
    parser.add_argument('--data_root', type=str)
    args = parser.parse_args()

    dataset = args.dataset
    data_dir = args.data_root
    feat_dir = os.path.join(data_dir, dataset)
    desc_name = args.desc_name

    if dataset == 'gldv2':
        combine(feat_dir, desc_name, None)

    if dataset.startswith(('gldv2-test', 'met')):
        combine(feat_dir, desc_name, 'gallery')

    elif dataset in ('roxford5k+1m', 'rparis6k+1m'):
        dtype, file_shape = combine(os.path.join(data_dir, 'revisitop1m'), desc_name, None)
        dataset = dataset.split('+')[0]
        with open(osp.join(data_dir, dataset, 'gallery.txt')) as fid:
            db_lines = fid.read().splitlines()

        r1m_lines = file_shape[0]
        hdf5_file = h5py.File(os.path.join(feat_dir, f'{desc_name}_gallery_local.hdf5'), 'w')
        hdf5_dataset = h5py.VirtualLayout(shape=((len(db_lines) + r1m_lines), *file_shape[1:]), dtype=dtype)

        vsource = h5py.VirtualSource(osp.join(data_dir, dataset, f'{desc_name}_gallery_local.hdf5'),
                                     'features', shape=(len(db_lines),*file_shape[1:]))
        hdf5_dataset[:len(db_lines)] = vsource

        vsource = h5py.VirtualSource(osp.join(data_dir, 'revisitop1m', f'{desc_name}_local.hdf5'),
                                     'features', shape=(r1m_lines,*file_shape[1:]))
        hdf5_dataset[len(db_lines):] = vsource

        hdf5_file.create_virtual_dataset('features', hdf5_dataset)
        hdf5_file.close()


if __name__ == '__main__':
    main()