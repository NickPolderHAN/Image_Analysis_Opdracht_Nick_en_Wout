from medmnist import ChestMNIST


def get_dataset():
    # Data ophalen en splitsen in train en test sets.
    dataset_train = ChestMNIST(split='train', download=True, size=128)
    dataset_test = ChestMNIST(split='test', download=True, size=128)

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
