from medmnist import ChestMNIST

def get_dataset():
    dataset_train = ChestMNIST(split='train', download=True, size=128)
    dataset_test = ChestMNIST(split='test', download=True, size=128)

    return dataset_train, dataset_test


if __name__ == '__main__':
    get_dataset()
