from topology.complex.base import Element, Complex


class Simplex(Element):
    def __init__(self, vertices, **data):
        Element.__init__(self, vertices, len(vertices)-1, **data)
    def key(self):
        return Element.key(self)[1]
    def __repr__(self):
        return '[%s]' % ' '.join(map(str, self))

class SimplicialComplex(Complex):
    def face_it(cls, s):
        dim = len(s)-1
        if dim:
            for i in range(dim+1):
                yield s[:i]+s[i+1:]
    def add_new(self, s, faces, **data):
        return self.add(Simplex(s, **data), faces)

# class DelaunayComplex(SimplicialComplex):
#     def __init__(self, P):
#         SimplicialComplex.__init__(self, P.shape[-1])
#         self.P = P
#         self.circumcenters = {}
#         delaunay = Delaunay(P)
#         for s in delaunay.simplices:
#             self.add_closure(stuple(s))
#     def add_closure(self, s):
#         if len(s) == 1:
#             return self.add_new(s, [], alpha=0)
#         c = self.get_circumcenter(s)
#         self.circumcenters[s] = c
#         alpha = la.norm(self.P[s[0]] - c) ** 2
#         faces = list(self.face_it(s))
#         for i,f in enumerate(faces):
#             if not f in self:
#                 self.add_closure(f)
#             if len(f) > 1:
#                 fc = self.circumcenters[f]
#                 if (la.norm(self.P[i] - fc) ** 2 < self[f]('alpha')
#                         and alpha < self[f]('alpha')):
#                     self[f].data['alpha'] = alpha
#         return self.add_new(s, faces, alpha=alpha)
#     def get_circumcenter(self, s):
#         dim, p = len(s)-1, self.P[list(s)]
#         return (tet_circumcenter(p) if dim == 3
#             else tri_circumcenter(p) if dim == 2
#             else p.sum(0) / 2 if dim == 1 else None)
#     # def get_alpha(self, s):
#     #     dim, p = len(s)-1, self.P[list(s)]
#     #     return (tet_circumradius(p) if dim == 3
#     #         else tri_circumradius(p) if dim == 2
#     #         else la.norm(diff(p)) if dim == 1
#     #         else 0) ** 2 / 4
