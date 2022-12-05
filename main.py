import math
import numpy as np
import multiprocessing as mp
import itertools as it

RADIUS = 1600


class MnistItem:
    """
    Class that covers one MnistItem (one row in dataset)
    """

    def __init__(self, id, label, values):
        """
        Constructor
        :param id: unique identificator
        :param label: number that this item represents
        :param values: coordinates
        """
        self.id = id
        self.label = label
        self.coordinates = values
        self.centroid = values

    def __str__(self):
        """
        To string function
        :return: str
        """
        return f"{self.label} - {self.coordinates}"


def load(file_name):
    """
    Function that loads file
    :param file_name: str filename
    :return: list
    """
    file = open(file_name, 'r')
    items = []
    id = 0

    for line in file.readlines()[1:]:
        item = list(map(int, line.replace('\n', '').split(',')))
        items.append(MnistItem(id, item[0], item[1:]))
        id += 1

    return items


def K(norm):
    """
    Kernel function
    :param norm: euclid distance
    :return:
    """
    return (1 / (RADIUS * math.sqrt(2 * math.pi))) * np.exp(-0.5 * (norm / RADIUS) ** 2)


def get_distance(x, y):
    """
    Euclid distance
    :param x: item1 coordinates
    :param y: item2 coordinates
    :return:
    """
    d = 0

    for a, b in zip(x, y):
        d += (a - b) * (a - b)

    return np.sqrt(d)


def mean_shift(items, item_key):
    """
    Main mean shift function
    :param items: all other items
    :param item_key: selected item key
    :return:
    """
    item = items[item_key]
    tolerance = 10

    while True:
        neighbours = []

        for another_item in items:
            # don't check the same item
            if another_item.id == item.id:
                continue

            # get euclid distance between items
            distance = get_distance(item.centroid, another_item.coordinates)

            # if another item is in selected radius add it to neighbours
            if distance <= RADIUS:
                neighbours.append(another_item)

        # if selected item does not have neighbours end the algorithm
        if len(neighbours) == 0:
            break

        # citatel
        numerator = np.zeros(len(item.coordinates))
        # menovatel
        denominator = 0.0
        for xi in neighbours:
            # euclid distance for kernel
            distance = get_distance(item.centroid, xi.coordinates)
            # kernel
            kernel = K(distance)
            # add to denominator
            denominator += kernel
            # add to numerator
            numerator += np.array(xi.coordinates) * kernel

        old_centroid = item.centroid
        item.centroid = list(map(int, np.divide(numerator, denominator)))

        # check if centroid moved (with some tolerance)
        if get_distance(item.centroid, old_centroid) < tolerance:
            break

    return item


if __name__ == '__main__':
    items = load('mnist_test.csv')
    # take only first 100 items (counting centroids from them)
    ids = list(range(0, 100))

    # parallel call
    with mp.Pool(processes=8) as pool:
        centroids = pool.starmap(mean_shift, zip(it.repeat(items), ids))

    # tolerance for clusters
    tolerance = 30
    clusters = []
    for centroid in centroids:
        cluster_found = False

        for cluster in clusters:
            if get_distance(cluster.centroid, centroid.centroid) < tolerance:
                cluster_found = True
                break

        if not cluster_found:
            clusters.append(centroid)

    print(f'Total number of clusters: {len(clusters)}')
