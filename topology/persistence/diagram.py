from topology.util import diff, identity

from functools import reduce
from tqdm import tqdm
import numpy as np

# TODO  row and column relative indices
#       unpairs contains death indices, pairs contains birth indices

class Reduction:
    __slots__ = ['__sequence', '__n', 'R', 'coh', 'dim','pivot_map',
                    'unpairs', 'pairs', 'copairs', 'D']
    def __init__(self, K, F, R, coh, pivot):
        self.__sequence, self.__n = F.get_range(R, coh), len(F) - len(R)
        self.R, self.coh, self.dim = R, coh, F.dim
        self.pivot_map = {i : F.index(s) for i,s in enumerate(pivot)}
        self.unpairs, self.pairs, self.copairs = set(self), {}, {}
        self.D = F.get_matrix(K, self, coh, pivot)
    def __iter__(self):
        for seq in self.__sequence:
            yield from seq
    def __len__(self):
        return self.__n
    def __setitem__(self, b, d):
        if self.coh:
            b, d = d, b
        self.pairs[b] = d
        self.copairs[d] = b
        # TODO (I)
        self.unpairs.remove(self.pivot_map[b])
        if d in self.unpairs:
            self.unpairs.remove(d)
    def __pair(self, low):
        pairs = self.copairs if self.coh else self.pairs
        return pairs[low] if low in pairs else None
    def __paired(self, low):
        return low is not None and low in (self.copairs if self.coh else self.pairs)
    def reduce(self, clearing=False, verbose=False, desc='[ persist'):
        for i in (tqdm(self, total=len(self), desc=desc) if verbose else self):
            if not (clearing and self.__paired(i)):
                low = self.D[i].get_pivot(self.R)#, self.pivot_map)
                while self.__paired(low):
                    self.D[i] += self.D[self.__pair(low)]
                    low = self.D[i].get_pivot(self.R)#, self.pivot_map)
                if low is not None:
                    self[low] = i

class Diagram(Reduction):
    __slots__ = Reduction.__slots__ + ['diagram', 'fmap']
    def __init__(self, K, F, R=set(), coh=False, pivot=None, clearing=False, verbose=False):
        pivot = F if pivot is None else pivot
        Reduction.__init__(self, K, F, R, coh, pivot)
        self.reduce(clearing, verbose)
        self.diagram, self.fmap = self.get_diagram(K, F, pivot)
    def __call__(self, i):
        if i in self.fmap:
            return self.fmap[i]
        return None
    def __getitem__(self, i):
        return self.pairs[i] if i in self.pairs else None
    def __contains__(self, i):
        return i is not None and i in self.pairs
    def items(self):
        yield from self.pairs.items()
    def keys(self):
        yield from self.pairs.keys()
    def values(self):
        yield from self.pairs.values()
    def is_relative(self, i):
        return i in self.R
    def get_diagram(self, K, F, pivot):
        fmap = {}
        dgms = [[] for d in range(self.dim+1)]
        for i, j in self.items():
            b, d = K[pivot[i]], K[F[self[i]]]
            fmap[i] = [b(pivot.key), d(F.key)][::(-1 if F.reverse else 1)]
            if fmap[i][0] < fmap[i][1]:
                dgms[b.dim].append(fmap[i])
        for i in self.unpairs:
            b = K[F[i]]
            fmap[i] = [b(pivot.key), np.inf]
            dgms[b.dim].append(fmap[i])
        dgms = list(map(np.array, dgms))
        return dgms, fmap
    # def element_pair(self, K, F, i):
    #     return [K[F[i]], K[F[self[i]]]]
    # def element_diagram(self, K, F):
    #     it = map(lambda i: self.element_pair(K, F, i), self)
    #     return partition(lambda L,p: insert(L, p[0].dim, p), it, self.dim+1)
    # def key_pair(self, K, key, p):
    #     return np.array([K[p[0]](key), np.inf if p[1] is None else K[p[1]](key)])
    # def get_diagram(self, K, F):
    #     f = lambda d: sorted(d, key=partial(self.key_pair, K, F.key))
    #     edgm = map(f, self.element_diagram(K, F)
    #     dgm = [np.vstack(map(partial(self.key_pair, K, F.key), d)) for d in edgm]
