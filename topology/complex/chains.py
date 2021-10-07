

class Column:
    __slots__ = ['elements', 'column', 'n', 'm']
    @classmethod
    def sum(cls, *columns):
        return sum(*columns, cls(set(),set()))
    def __init__(self, elements, column):
        self.elements, self.column = elements, column
        self.n, self.m = len(self.elements), len(self.column)
    def get_pivot(self, relative=set()):
        if self.m == 0:
            return None
        if not relative:
            return self.column[-1]
        for i in self.column[::-1]:
            if not i in relative:
                return i
        return None
    def _add_column(self, other):
        column = []
        i, j = 0, 0
        while i < self.m and j < other.m:
            if i < self.m and self._cmp(self.column[i], other.column[j]):
                column += [self.column[i]]
                i += 1
                while i < self.m and self._cmp(self.column[i], other.column[j]):
                    column += [self.column[i]]
                    i += 1
            elif j < other.m and other._cmp(other.column[j], self.column[i]):
                column += [other.column[j]]
                j += 1
                while j < other.m and other._cmp(other.column[j], self.column[i]):
                    column += [other.column[j]]
                    j += 1
            else:
                i += 1
                j += 1
        if i < self.m:
            column += self.column[i:]
        elif j < other.m:
            column += other.column[j:]
        return column
    def _add_elements(self, other):
        return self.elements ^ other.elements
    def __add__(self, other):
        return type(self)(self._add_elements(other), self._add_column(other))
    def __len__(self):
        return self.n
    def __iter__(self):
        yield from self.elements
    def __repr__(self):
        return '+'.join([str(s) for s in self])

class Chain(Column):
    def _cmp(self, a, b):
        return a < b
    def __repr__(self):
        return 'Chain(%s)' % Column.__repr__(self)

class CoChain(Column):
    def _cmp(self, a, b):
        return a > b
    def __repr__(self):
        return 'CoChain(%s)' % Column.__repr__(self)
