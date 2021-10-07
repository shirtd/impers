from topology.complex.base import ComplexElement, Complex

import diode


class STNode:
    def __init__(self, label, parent=None):
        self.label, self.parent = label, parent
        self.children = {}
        if parent is not None:
            parent.add_child(self)
    def get_children(self):
        return list(self.children.values())
    def get_child(self, key):
        if key in self.children:
            return self.children[key]
    def add_child(self, other):
        self.children[other.label] = other
    def __eq__(self, other):
        return (self.label == other.label
            and (self.parent is None
                or self.parent == other.parent))

class STSimplex(ComplexElement, STNode):
    def __init__(self, label, parent=None, **data):
        dim = 0 if parent is None else parent.dim + 1
        ComplexElement.__init__(self, dim, **data)
        STNode.__init__(self, label, parent)
    def astuple(self):
        return tuple(self)
    def get_simplex(self):
        if self.dim == 0:
            return (self.label,)
        return self.parent.get_simplex() + (self.label,)
    def is_simplex(self, s):
        if self.label == s[-1]:
            return (not self.dim
                or self.parent.is_simplex(s[:-1]))
        return False
    def __iter__(self):
        if self.parent is not None:
            yield from self.parent
        yield self.label
    def is_coface(self, s):
        if s.dim > self.dim:
            return False
        elif s.label == self.label:
            return not (self.dim and s.dim) or self.parent.is_coface(s.parent)
        elif self.dim and s.label == self.parent.label:
            return self.parent.is_coface(s)
    def __getitem__(self, i):
        if i == self.dim:
            return self.label
        return self.parent[i]
    def __eq__(self, other):
        return (ComplexElement.__eq__(self, other)
            and STNode.__eq__(self, other))
    def __repr__(self):
        return '[%s]' % ' '.join(map(str,self))
    def __len__(self):
        return self.dim+1
    def __lt__(self, other):
        return (self.dim, tuple(self)) < (other.dim, tuple(other))
    def get_post(self, post=None):
        if post is None or not len(post):
            return self
        elif post[0] in self.children:
            c = self.get_child(post[0])
            if len(post) == 1:
                return c
            return c.get_post(post[1:])


class STComplex(Complex):
    def __init__(self, dim):
        Complex.__init__(self, dim)
        self.tree = {d : {} for d in range(dim+1)}
    def root(self, v):
        if v in self.tree[0]:
            return self.tree[0][v][0]
    def prefix(self, s, p=None):
        if len(s) > 1:
            p = self.root(s[0]) if p is None else p.get_child(s[0])
            return None if p is None else self.prefix(s[1:], p)
        return p
    def add_new(self, s, **data):
        return self.add(STSimplex(s[-1], self.prefix(s), **data))
    def add(self, s):
        if not s.label in self.tree[s.dim]:
            self.tree[s.dim][s.label] = [s]
        else:
            self.tree[s.dim][s.label].append(s)
        return s
    def get_cofaces(self, s):
        s = s if isinstance(s, Simplex) else self[s]
        cofaces = s.get_children()
        if s.dim+1 in self.tree and s.label in self.tree[s.dim+1]:
            cofaces += [c for c in self.tree[s.dim+1][s.label] if c.is_coface(s)]
        return cofaces
    def get_faces(self, s, post=None):
        if not isinstance(s, Simplex):
            s = self[s]
        if s.parent is None:
            if post is None:
                return []
            return [self[post]]
        f = s.parent.get_post(post)
        post = [s.label] + ([] if post is None else post)
        return [f] + self.get_faces(s.parent, post)
    def __getitem__(self, s):
        if len(s) > 1:
            return self.prefix(s).get_child(s[-1])
        return self.root(s[0])
    def __iter__(self):
        for d,S in self.tree.items():
            for s in S.values():
                yield from s
    def __call__(self, dim):
        return [s for S in self.tree[dim].values() for s in S]
    def __contains__(self, s):
        p = self.prefix(key)
        return p is not None and s[-1] in p.children

class AlphaSTComplex(STComplex):
    def __init__(self, P):
        STComplex.__init__(self, P.shape[-1])
        self.P = P
        for s,f in diode.fill_alpha_shapes(P, True):
            self.add_new(sorted(s), alpha=f)
