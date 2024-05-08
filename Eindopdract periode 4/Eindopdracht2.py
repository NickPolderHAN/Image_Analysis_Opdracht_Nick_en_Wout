from medmnist import ChestMNIST

dataset_train = ChestMNIST(split='train', download=True, size=128)
dataset_test = ChestMNIST(split='test', download=True, size=128)
print(dataset_train, dataset_test)
