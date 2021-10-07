from topology.util import stuple


class Element:
    __slots__ = ['dim', 'data']
    def __init__(self, atoms, dim, **data):
        self.dim, self.data = dim, data
        self.__atoms = stuple(atoms)
    def __iter__(self):
        yield from self.__atoms
    def __len__(self):
        return len(self.__atoms)
    def __index__(self):
        return list(self.__atoms)
    def __getitem__(self, i):
        return self.__atoms[i]
    def __call__(self, key):
        if key in self.data:
            return self.data[key]
    def key(self):
        return (self.dim, self.__atoms)
    def __hash__(self):
        return hash(self.key())
    def __eq__(self, other):
        return hash(self) == hash(other)
    def __lt__(self, other):
        return (self.dim, self.__atoms) < (other.dim, other.__atoms)

class Complex:
    __slots__ = ['dim']
    def __init__(self, dim):
        self.dim = dim
        self.__elements = {d : [] for d in range(dim+1)}
        self.__map, self.__faces, self.__cofaces = {}, {}, {}
    def add(self, cell, faces):
        k = hash(cell)
        if k in self:
            return cell
        self.__elements[cell.dim].append(cell)
        self.__map[k] = cell
        self.__faces[k] = set()
        self.__cofaces[k] = set()
        for f in faces:
            fk = hash(f)
            self.__faces[k].add(fk)
            self.__cofaces[fk].add(k)
        return cell
    def faces(self, c):
        return [self.__map[f] for f in self.__faces[hash(c)]]
    def cofaces(self, c):
        return [self.__map[f] for f in self.__cofaces[hash(c)]]
    def items(self):
        yield from self.__elements.items()
    def keys(self):
        yield from self.__elements.keys()
    def values(self):
        yield from self.__elements.values()
    def __getitem__(self, key):
        if isinstance(key, int):
            key = (key,)
        return self.__map[hash(key)]
    def __call__(self, dim):
        if dim in self.__elements:
            return self.__elements[dim]
        return set()
    def __len__(self):
        return len(self.__map)
    def __iter__(self):
        yield from self.__map.values()
    def __contains__(self, key):
        return hash(key) in self.__map
    def get_sequence(self, key, reverse=False):
        r = -1 if reverse else 1
        return sorted(self, key=lambda s: (r * s(key), s))
    def closure(self, s):
        return {s}.union({f for t in self.faces(s) for f in self.closure(t)})
    def __repr__(self):
        return ''.join(['%d:\t%d elements\n' % (d, len(self(d))) for d in range(self.dim+1)])
