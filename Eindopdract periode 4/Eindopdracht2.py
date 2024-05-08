from medmnist import ChestMNIST
import tensorflow


def get_dataset():
    # The pictures and labels for the train set.
    pictures_train = []
    labels_train = []

    # The pictures and labels for the test set.
    pictures_test = []
    labels_test = []

    # Data ophalen en splitsen in train en test sets.
    dataset_train = ChestMNIST(split='train', download=True, size=128)
    dataset_test = ChestMNIST(split='test', download=True, size=128)

    # Amount of labels.
    label_count = 14

    # The training set.
    for i in dataset_train:
        if 1 in i[1]:
            pictures_train.append(i[0])
            labels_train.append(1)
        else:
            pictures_train.append(i[0])
            labels_train.append(0)

    # The test set.
    for i in dataset_test:
        if 1 in i[1]:
            pictures_test.append(i[0])
            labels_test.append(1)
        else:
            pictures_test.append(i[0])
            labels_test.append(0)

    return dataset_train, dataset_test


if __name__ == '__main__':
    get_dataset()


# De images worden correct voorbewerkt.
# Model wordt bewaard en is beschikbaar voor toekomstig gebruik.
# Het classificatie model is correct gegenereerd.
# Splisting van data is efficient en correct
# Classificatieperformance is acceptabel (>80% correct)
# Validatie van het classificatiemodel is aantoonbaar correct
# Preventie van overfitting is correct toegepast.
