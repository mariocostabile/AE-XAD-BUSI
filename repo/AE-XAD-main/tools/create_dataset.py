import os

import PIL.Image as Image
import numpy as np

def mvtec(cl, path, n_anom_per_cls, seed=None):
    '''
    Method to arrange the data. Train and test images are supposed to be stored in two different folders.
    The train folder consists of only normal images stored in the 'good' subfolder. The test folder contains normal data
    in the 'good' subfolder and anomalous data into different subfolders representing different types of anomalies.
    A number of anomalies equals to n_anom_per_cls is taken from the test folder and included into the training set.
    :param cl: int, number of object class to consider
    :param path: str, path in which the images are stored
    :param n_anom_per_cls: int, number of anomalies to be included into the training set
    :param seed: seed to use for reproducibility, if None a random seed is selected, defaults to None
    '''
    np.random.seed(seed=seed)

    labels = (
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper'
    )

    class_o = labels[cl]

    root = os.path.join(path, class_o)

    X_train = []
    X_test = []
    GT_train = []
    GT_test = []

    # Add normal data to train set
    f_path = os.path.join(root, 'train', 'good')
    normal_files_tr = os.listdir(f_path)
    for file in normal_files_tr:
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
            image = np.array(Image.open(os.path.join(f_path, file)).convert('RGB'))
            X_train.append(image)
            GT_train.append(np.zeros_like(image, dtype=np.uint8))

    # Add normal data to test set
    f_path = os.path.join(root, 'test', 'good')
    normal_files_te = os.listdir(f_path)
    for file in normal_files_te:
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
            image = np.array(Image.open(os.path.join(f_path, file)).convert('RGB'))
            X_test.append(image)
            GT_test.append(np.zeros_like(image, dtype=np.uint8))

    outlier_data_dir = os.path.join(root, 'test')
    outlier_classes = os.listdir(outlier_data_dir)
    for cl_a in outlier_classes:
        if cl_a == 'good':
            continue

        outlier_file = np.array(os.listdir(os.path.join(outlier_data_dir, cl_a)))
        idxs = np.random.permutation(len(outlier_file))

        # Train
        for file in outlier_file[idxs[: n_anom_per_cls]]:
            print(file)
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                X_train.append(np.array(Image.open(os.path.join(root, 'test/' + cl_a + '/' + file)).convert('RGB')))
                GT_train.append(np.array(Image.open(os.path.join(root, 'ground_truth/' + cl_a + '/' + file).replace('.png', '_mask.png')).convert('RGB')))

        # Test
        for file in outlier_file[idxs[n_anom_per_cls:]]:
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                X_test.append(np.array(Image.open(os.path.join(root, 'test/' + cl_a + '/' + file)).convert('RGB')))
                GT_test.append(np.array(Image.open(os.path.join(root, 'ground_truth/' + cl_a + '/' + file).replace('.png', '_mask.png')).convert('RGB')))

    print('GT ', len(GT_train))
    X_train = np.array(X_train).astype(np.uint8)

    X_test = np.array(X_test).astype(np.uint8)

    print(X_train.shape)
    print(X_test.shape)

    GT_train = np.array(GT_train)

    GT_test = np.array(GT_test)


    Y_train = np.zeros(X_train.shape[0])
    Y_train[len(normal_files_tr): ] = 1
    Y_test = np.zeros(X_test.shape[0])
    Y_test[len(normal_files_te): ] = 1

    print(Y_train.sum())
    print(Y_test.sum())

    return X_train, Y_train, X_test, Y_test, GT_train, GT_test


def extract_dataset(path, n_anom_per_cls, seed=None):
    '''
    Method to arrange the data. Train and test images are supposed to be stored in two different folders.
    The train folder consists of only normal images stored in the 'good' subfolder. The test folder contains normal data
    in the 'good' subfolder and anomalous data into different subfolders representing different types of anomalies.
    A number of anomalies equals to n_anom_per_cls is taken from the test folder and included into the training set.
    :param path: str, path in which the dataset is stored
    :param n_anom_per_cls: int, number of anomalies to be included into the training set
    :param seed: seed to use for reproducibility, if None a random seed is selected, defaults to None
    '''
    np.random.seed(seed=seed)

    X_train = []
    X_test = []
    GT_train = []
    GT_test = []

    # Add normal data to train set
    f_path = os.path.join(path, 'train', 'good')
    normal_files_tr = os.listdir(f_path)
    for file in normal_files_tr:
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:] or 'bmp' in file[-3:]:
            image = np.array(Image.open(os.path.join(f_path, file)).convert('RGB'))
            X_train.append(image)
            GT_train.append(np.zeros_like(image, dtype=np.uint8))

    # Add normal data to test set
    f_path = os.path.join(path, 'test', 'good')
    normal_files_te = os.listdir(f_path)
    for file in normal_files_te:
        if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:] or 'bmp' in file[-3:]:
            image = np.array(Image.open(os.path.join(f_path, file)).convert('RGB'))
            X_test.append(image)
            GT_test.append(np.zeros_like(image, dtype=np.uint8))

    outlier_data_dir = os.path.join(path, 'test')
    outlier_classes = os.listdir(outlier_data_dir)
    for cl_a in outlier_classes:
        if cl_a == 'good':
            continue

        outlier_file = np.array(os.listdir(os.path.join(outlier_data_dir, cl_a)))
        idxs = np.random.permutation(len(outlier_file))

        # Train
        for file in outlier_file[idxs[: n_anom_per_cls]]:
            print(file)
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:] or 'bmp' in file[-3:]:
                X_train.append(np.array(Image.open(os.path.join(path, 'test/' + cl_a + '/' + file)).convert('RGB')))
                GT_train.append(np.array(Image.open(os.path.join(path, 'ground_truth/' + cl_a + '/' + file).replace(f'.{file[-3:]}', '.png')).convert('RGB')))

        # Test
        for file in outlier_file[idxs[n_anom_per_cls:]]:
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:] or 'bmp' in file[-3:]:
                X_test.append(np.array(Image.open(os.path.join(path, 'test/' + cl_a + '/' + file)).convert('RGB')))
                GT_test.append(np.array(Image.open(os.path.join(path, 'ground_truth/' + cl_a + '/' + file).replace(f'.{file[-3:]}', '.png')).convert('RGB')))

    print('GT ', len(GT_train))
    X_train = np.array(X_train).astype(np.uint8)

    X_test = np.array(X_test).astype(np.uint8)

    print(X_train.shape)
    print(X_test.shape)

    GT_train = np.array(GT_train)

    GT_test = np.array(GT_test)


    Y_train = np.zeros(X_train.shape[0])
    Y_train[len(normal_files_tr): ] = 1
    Y_test = np.zeros(X_test.shape[0])
    Y_test[len(normal_files_te): ] = 1

    print(f'Training anomalies: {Y_train.sum()}')
    print(f'Test anomalies: {Y_test.sum()}')

    return X_train, Y_train, X_test, Y_test, GT_train, GT_test
