from topology.complex.chains import Chain, CoChain
from topology.util import insert, partition


class Filtration:
    __slots__ = ['sequence', 'dim', 'key', 'reverse', 'imap']
    def __init__(self, K, key, reverse=False):
        self.sequence = K.get_sequence(key, reverse)
        self.dim, self.key, self.reverse = K.dim, key, reverse
        self.imap = {hash(s) : i for i, s in enumerate(self)}
    def __len__(self):
        return len(self.sequence)
    def __iter__(self):
        yield from self.sequence
    def __getitem__(self, i):
        return self.sequence[i]
    def index(self, s):
        return self.imap[hash(s)]
    def get_range(self, R=set(), coh=False):
        it = reversed(list(enumerate(self))) if coh else enumerate(self)
        f = lambda L,ix: L if ix[0] in R else insert(L, ix[1].dim, ix[0])
        return partition(f, it, self.dim+1)[::(1 if coh else -1)]
    def sort_faces(self, K, i):
        return sorted([self.index(f) for f in K.faces(self[i])])
    def sort_cofaces(self, K, i):
        return sorted([self.index(f) for f in K.cofaces(self[i])], reverse=True)
    def as_chain(self, K, i):
        return Chain({i}, self.sort_faces(K, i))
    def as_cochain(self, K, i):
        return CoChain({i}, self.sort_cofaces(K, i))
    def get_chains(self, K, rng):
        return {i : self.as_chain(K, i) for i in rng}
    def get_cochains(self, K, rng):
        return {i : self.as_cochain(K, i) for i in rng}
    def get_matrix(self, K, rng, coh=False):
        return (self.get_cochains if coh else self.get_chains)(K, rng)
