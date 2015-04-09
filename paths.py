import sys
import vigra
import numpy
import os
import matplotlib.pyplot as plt


def parse_image(filename, tile_x, tile_y):
    """Load the given image and parse it on a (tile_x, tile_y) grid.

    :param filename: image filename
    :param tile_x: number of tiles in x direction
    :param tile_y: number of tiles in y direction
    :return: parsed image as numpy array
    """
    out = numpy.zeros((tile_x, tile_y))
    data = vigra.readImage(filename)
    dx = data.shape[0] / float(tile_x)
    dy = data.shape[1] / float(tile_y)
    for y in xrange(tile_y):
        for x in xrange(tile_y):
            scanx = x * dx + dx/2
            scany = y * dy + dy/2
            if max(data[scanx, scany, :]) < 70:
                out[x, y] = 0
            else:
                out[x, y] = 4
    return out


def dist_direct(a, b):
    """Returns sum of the pairwise distances of a and b.
    """
    return numpy.sum(numpy.abs(numpy.array(b)-numpy.array(a)))


def dist_euclidean(a, b):
    """Returns the euclidean dist of a and b.
    """
    return numpy.linalg.norm(numpy.array(b)-numpy.array(a))


def dist_zero(a, b):
    """Returns zero.
    """
    return 0


def neighbors4(pnt, data):
    """Return the 4 neighbors of a given point from an array.

    :param pnt: point
    :param data: data, walls have value 0
    :return: list with neighbor points
    """
    assert len(pnt) == 2
    assert len(data.shape) == 2
    tmp_neighbors = []
    if pnt[0] > 0:
        tmp_neighbors.append((pnt[0]-1, pnt[1]))
    if pnt[0]+1 < data.shape[0]:
        tmp_neighbors.append((pnt[0]+1, pnt[1]))
    if pnt[1] > 0:
        tmp_neighbors.append((pnt[0], pnt[1]-1))
    if pnt[1]+1 < data.shape[1]:
        tmp_neighbors.append((pnt[0], pnt[1]+1))
    neighbors = []
    for p in tmp_neighbors:
        if data[p] != 0:
            neighbors.append(p)
    return neighbors


def neighbors8(pnt, data):
    """Return the 8 neighbors of a given point from an array.

    :param pnt: point
    :param data: data, walls have value 0
    :return: list with neighbor points
    """
    assert len(pnt) == 2
    assert len(data.shape) == 2
    sh = data.shape
    neighbors = []
    for y in range(3):
        for x in range(3):
            new_pnt = (pnt[0]+x-1, pnt[1]+y-1)
            if 0 <= new_pnt[0] < sh[0] and 0 <= new_pnt[1] < sh[1] and not (x == 1 and y == 1) and data[new_pnt] != 0:
                neighbors.append(new_pnt)
    return neighbors


class DefaultVisitor(object):
    """Visitor that does nothing.
    """

    def visit(self, other):
        pass

    def endvisit(self, other):
        pass


class DrawVisitor(object):
    """Visitor that draws an animation.
    """

    def __init__(self, data):
        self.data = data
        self.tmpdata = numpy.array(data)
        self.oldopen = []
        plt.ion()
        self.ax = plt.gca()
        self.fig = plt.gcf()
        self.im = self.ax.imshow(numpy.swapaxes(self.data, 0, 1), interpolation="nearest")
        plt.draw()

    def beginvisit(self, other):
        self.tmpdata[other.start] = 3
        self.tmpdata[other.goal] = 3
        self.im.set_data(numpy.swapaxes(self.tmpdata, 0, 1))
        plt.draw()

    def visit(self, other):
        for p in self.oldopen:
            self.tmpdata[p] = 1
        self.oldopen = list(other.openitems())
        for p in self.oldopen:
            self.tmpdata[p] = 2
        self.im.set_data(numpy.swapaxes(self.tmpdata, 0, 1))
        plt.draw()

    def endvisit(self, other):
        for p in other.path():
            self.tmpdata[p] = 3
        self.im.set_data(numpy.swapaxes(self.tmpdata, 0, 1))
        plt.draw()
        plt.ioff()
        plt.show()


class PQueue(object):
    """Priority queue that uses a list and bisect.bisect_right.
    """

    def __init__(self, scorer):
        self.scorer = scorer
        self.items = []
        self.item_scores = []

    def put(self, item):
        import bisect
        score = self.scorer[item]
        i = bisect.bisect_right(self.item_scores, score)
        self.items.insert(i, item)
        self.item_scores.insert(i, score)

    def get(self):
        return self.items[0]

    def pop(self):
        self.items.pop(0)
        self.item_scores.pop(0)

    def contains(self, item):
        return item in self.items

    def empty(self):
        return len(self.items) == 0


class AStar(object):
    """AStar path finder algorithm.
    """

    def __init__(self, data):
        self.data = data
        self.came_from = {}
        self.start = None
        self.goal = None
        self.closedset = []
        self.openqueue = None

    def run(self, start, goal, heuristic=dist_euclidean, visitor=None, neighborfunc=neighbors8):
        if visitor is None:
            visitor = DefaultVisitor()

        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        self.start = start
        self.goal = goal
        self.closedset = []
        self.openqueue = PQueue(f_score)
        self.openqueue.put(start)
        self.came_from = {}

        visitor.beginvisit(self)
        while not self.openqueue.empty():
            current = self.openqueue.get()
            if current == self.goal:
                break

            self.openqueue.pop()
            self.closedset.append(current)
            for p in neighborfunc(current, self.data):
                if p in self.closedset:
                    continue
                tentative_g_score = g_score[current] + dist_euclidean(current, p)

                if not self.openqueue.contains(p) or tentative_g_score < g_score[p]:
                    self.came_from[p] = current
                    g_score[p] = tentative_g_score
                    f_score[p] = g_score[p] + heuristic(p, self.goal)
                    if not self.openqueue.contains(p):
                        self.openqueue.put(p)

            visitor.visit(self)
        visitor.endvisit(self)

    def openitems(self):
        return self.openqueue.items

    def path(self):
        node = self.goal
        nodes = [node]
        while node in self.came_from:
            node = self.came_from[node]
            nodes.append(node)
        return nodes


def main():
    if os.path.isfile("data.npy"):
        data = numpy.load("data.npy")
        print "loaded from data.npy"
    elif os.path.isfile("raw.png"):
        data = parse_image("raw.png", 100, 100)
        numpy.save("data.npy", data)
        print "loaded from raw.png"
    else:
        print "Could not find data.npy and could not find raw.png."
        sys.exit(1)

    pfrom = (35, 99)
    pto = (56, 2)
    pathfinder = AStar(data)
    vis = DrawVisitor(data)
    pathfinder.run(pfrom, pto, heuristic=dist_euclidean, neighborfunc=neighbors8, visitor=vis)
    # pathfinder.run(pfrom, pto, heuristic=dist_euclidean, neighborfunc=neighbors4, visitor=vis)
    # pathfinder.run(pfrom, pto, heuristic=dist_direct, neighborfunc=neighbors4, visitor=vis)
    # pathfinder.run(pfrom, pto, heuristic=dist_zero, neighborfunc=neighbors8, visitor=vis)

    return 0


if __name__ == "__main__":
    status = main()
    sys.exit(status)
