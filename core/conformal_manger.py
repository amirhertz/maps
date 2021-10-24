from __future__ import annotations
from custom_types import *
from utils import mesh_utils
from core.decimate import Decimate


def point_in_triangle(vs: T, triangles: T, front_side=True):

    def sign(ind):
        d_vs_ = d_vs[:, :, (ind + 1) % 3]
        return d_vs_[:, :, 0] * d_t[ind][:, :, 1] - d_vs_[:, :, 1] * d_t[ind][:, :, 0]

    if triangles.dim() == 3:
        triangles = triangles.unsqueeze(0).expand(vs.shape[0], *triangles.shape)

    d_vs = vs[:, None, None] - triangles[:, :, :]
    d_t = [triangles[:, :, i] - triangles[:, :, (i + 1) % 3] for i in range(3)]
    signs = torch.stack([sign(i) for i in range(3)], 2)
    if front_side:
        in_triangle = torch.eq(torch.ge(signs, 0).sum(2), 3)
    else:
        in_triangle = torch.eq(torch.le(signs, 0).sum(2), 3)
    check = in_triangle.sum(1)
    on_edge_vs_inds = torch.where(check.ge(2))[0]
    if on_edge_vs_inds.shape[0] != 0:
        on_edge_hold_inds = in_triangle[on_edge_vs_inds].long().argmax(1)
        in_triangle[on_edge_vs_inds] = False
        in_triangle[on_edge_vs_inds, on_edge_hold_inds] = True
    boundary_vs_inds = torch.where(check.eq(0))[0]
    if boundary_vs_inds.shape[0] != 0:
        signs[boundary_vs_inds] = torch.relu(-signs[boundary_vs_inds])
        on_edge_hold_inds = signs[boundary_vs_inds].sum(2).argmin(1)
        in_triangle[boundary_vs_inds, on_edge_hold_inds] = True
    triangle_ind = torch.where(in_triangle)
    triangle_weights = mesh_utils.find_barycentric(vs, triangles[triangle_ind])
    # to simplex
    triangle_weights = triangle_weights / triangle_weights.sum(1)[:, None]
    return triangle_ind, triangle_weights


# def plot_uv(mesh: T_Mesh, uv: T, path: str):
#     vs, faces = mesh
#     vs, faces, uv = tuple(map(lambda x: x.numpy(), [vs, faces, uv]))
#     p = mp.plot(vs, faces, shading={"wireframe": True, "flat": False}, return_plot=True)
#     p.save(files_utils.add_suffix(f'{path}_3d', '.html'))
#     p = mp.plot(uv, faces, shading={"wireframe": True, "flat": False}, return_plot=True)
#     p.save(files_utils.add_suffix(f'{path}_2d', '.html'))


class ConformalMap:

    @staticmethod
    def cum_sum_with_zero(deg: T) -> T:
        cum_sum = torch.cumsum(deg, dim=0)[: -1]
        cum_sum = torch.cat((torch.zeros(1, dtype=cum_sum.dtype, device=cum_sum.device), cum_sum), dim=0)
        return cum_sum

    # from 3d faces_inds to uv_b vs_faces
    def map_reduced_uv(self, faces: T, searched_rings: T):
        faces_vs_inds = self.reduced_mesh[1][faces]
        vs_inds = self.vs_map[searched_rings]
        look_up = torch.eq(faces_vs_inds[:, :, None], vs_inds[:, None, :])
        look_up = torch.where(look_up)[2].view(look_up.shape[:-1])
        look_up = look_up + self.cum_sum_rings[searched_rings][:, None]
        return look_up

    #  from faces_inds, weights to uvb
    def uv2uv(self, indices, weights, searched_rings):
        vs_b = torch.einsum('rnd,rn->rd', self.uv_b[0][indices], weights)
        look_up_rings_faces = self.uv_a[1][searched_rings]
        look_up_rings_vs = self.uv_a[0][look_up_rings_faces]
        triangle_ind, triangle_weights = point_in_triangle(vs_b, look_up_rings_vs)
        return triangle_ind, triangle_weights

    def uv23d(self, uv_map, searched_rings):
        faces_map, rolling_map = self.back_face_mapper
        target_faces_b = faces_map[searched_rings][uv_map[0]]
        rolling_faces_b = rolling_map[searched_rings][uv_map[0]]
        rolling_weights = torch.gather(uv_map[1], 1, rolling_faces_b)
        # vs, faces = self.vs_a, self.uv_a[1]
        # target_faces = faces[searched_rings][uv_map[0]]
        # target_vs = self.target_mesh[0][vs[target_faces]]
        # target_vs = (target_vs * uv_map[1][:, :, None]).sum(1)
        # target_vs_b = self.target_mesh[0][self.target_mesh[1][target_faces_b]]
        # target_vs_b = (target_vs_b * rolling_weights[:, :, None]).sum(1)
        # assert not find_diif(target_vs, target_vs_b)[1]
        return target_faces_b, rolling_weights

    def easy_map(self, faces_inds, weights):
        faces = self.reduced_mesh[1][faces_inds]
        faces = self.back_mapper_vs[faces]
        vs = (self.target_mesh[0][faces] * weights[:, :, None]).sum(1)
        return vs

    # # in place
    # def for_debug(self, faces_inds: T, weights: T) -> TS:
    #     all_vs = torch.zeros(*weights.shape, device=faces_inds.device)
    #     searched_rings = self.faces_map[faces_inds]
    #     relevant_rings = searched_rings.gt(-1)
    #     searched_rings = searched_rings[relevant_rings]
    #     look_up = self.map_reduced_uv(faces_inds[relevant_rings], searched_rings)
    #     uv_map, vs_a, vs_b = self.uv2uv(look_up, weights[relevant_rings], searched_rings)
    #     colors = torch.rand(vs_b.shape[0], 3)
    #     files_utils.export_mesh(self.uv_b, "uv_b")
    #     files_utils.export_mesh(self.uv_a_raw, "uv_a")
    #     files_utils.export_mesh(vs_b, "vs_b", colors=colors)
    #     files_utils.export_mesh(vs_a, "vs_a", colors=colors)
    #     return all_vs


    # in place
    def upsample(self, faces_inds: T, weights: T) -> TS:
        searched_rings = self.faces_map[faces_inds]
        relevant_rings = searched_rings.gt(-1)
        if relevant_rings.any() > 0:
            searched_rings = searched_rings[relevant_rings]
            look_up = self.map_reduced_uv(faces_inds[relevant_rings], searched_rings)
            uv_map = self.uv2uv(look_up, weights[relevant_rings], searched_rings)
            faces_inds[relevant_rings], weights[relevant_rings] = self.uv23d(uv_map, searched_rings)
        faces_inds[~relevant_rings] = self.back_mapper[faces_inds[~relevant_rings]]
        # all_vs = torch.zeros(*weights.shape, device=faces_inds.device)
        # all_vs[relevant_rings] = self.uv23d(uv_map, searched_rings)
        # all_vs[~relevant_rings] = self.easy_map(faces_inds[~relevant_rings], weights[~relevant_rings])
        # return all_vs
        return faces_inds, weights

    def __call__(self, faces_inds: T, weights: T, in_place=True) -> TS:
        if not in_place:
            faces_inds, weights = faces_inds.clone(), weights.clone()
        return self.upsample(faces_inds, weights)

    @staticmethod
    def create_merged_rings(vs: T, faces: T, mask: T, cum_sum_a: T, updated_positions: T) -> TS:
        vs, faces, mask = vs.clone(), faces.clone(), mask.clone()
        boundary_indices = torch.zeros(vs.shape[0], vs.shape[1], dtype=torch.int64)
        boundary_indices += (torch.arange(vs.shape[1]) - 1)[None, :]
        cum_sum_b = cum_sum_a - torch.arange(cum_sum_a.shape[0])
        vs[:, 0] = updated_positions
        mask[:, 1] = False
        # mask = mask[:num_conformals]
        vs = vs[mask]
        mask[:, 0] = False
        faces = faces - 1
        faces[torch.eq(faces, -1)] = 0
        faces += cum_sum_b[:, None, None]
        boundary_select = boundary_indices + 1
        boundary_indices += cum_sum_b[:, None]
        boundary_select += cum_sum_a[:, None]
        return vs, faces[mask], boundary_indices[mask], boundary_select[mask], cum_sum_b.clone()

    def complete_map(self,  mesh: T_Mesh, vs_mapper: T, faces_mask: T):
        if self.in_low_phase:
            self.vs_map[:, 0] = self.vs_map[:, 2]
        else:
            self.vs_map[:, 1] = self.vs_map[:, 0]
        self.vs_map = self.vs_map[:, 1:]
        self.faces_map = self.faces_map[faces_mask]
        ma = torch.ne(self.vs_map, -1)
        self.vs_map[ma] = vs_mapper[self.vs_map[ma]]
        self.reduced_mesh = mesh
        self.back_mapper_vs = torch.where(torch.ge(vs_mapper, 0))[0]
        self.back_mapper = torch.where(faces_mask)[0]

    @staticmethod
    def create_faces_map(ds: Decimate) -> T:
        edge_key = ds.queue
        faces_map = torch.zeros(ds.faces.shape[0] + 1, dtype=torch.int64) - 1
        if ds.in_low_phase:
            vs_faces = ds.v2f[edge_key].view(edge_key.shape[0], -1)
        else:
            vs_faces = ds.v2f[ds.edges[edge_key]].view(edge_key.shape[0], -1)
        faces_map[vs_faces] = torch.arange(edge_key.shape[0])[:, None]
        faces_map = faces_map[:-1]
        return faces_map

    def separate_uv(self, uv):
        vs, faces = uv
        if self.in_low_phase:
            return vs, faces.view(self.num_rings, -1, 3)
        cum_sum = self.cum_sum_rings + torch.arange(self.cum_sum_rings.shape[0])
        cum_sum = torch.cat((cum_sum, torch.tensor([faces.shape[0]])), dim=0)
        deg = cum_sum[1:] - cum_sum[:-1]
        select = torch.arange(deg.max().item())
        select = select[None, :] + cum_sum[:-1, None]
        mask = torch.ge(select, cum_sum[1:, None])
        select[mask] = 0
        faces = faces[select]
        faces[0, mask[0]] = faces[-1, 0][None, :]
        separated_uv = vs, faces
        return separated_uv

    @staticmethod
    def pad_faces(faces: T) -> TS:
        ring_ma = torch.ne(faces, -1)
        masked = ring_ma.sum(1)
        pad_length = (masked.max() - masked.min()).item()
        to_mask = masked.max() - masked
        padding = to_mask[:, None] - torch.arange(pad_length, device=faces.device).unsqueeze(0).expand(faces.shape[0],
                                                                                                       pad_length) - 1
        padding = torch.ge(padding, 0)
        ring_ma = torch.cat((ring_ma, padding), 1)
        ring = torch.cat((faces, torch.zeros(padding.shape, dtype=faces.dtype, device=faces.device) - 1), 1)
        ring = ring[ring_ma].view(ring.shape[0], -1)
        mask = torch.ne(ring, -1)
        # degree = mask.sum(1)
        return ring, mask

    @staticmethod
    def create_faces_rings(ds: Decimate) -> TS:
        if ds.in_low_phase:
            faces_inds = ds.v2f[ds.queue][:, :3]
            faces_inds_mask = torch.ones_like(faces_inds, dtype=torch.bool)
        else:
            edge_key = ds.queue
            faces_inds = ds.v2f[ds.edges[edge_key]]  # rings X 2 X deg
            equal_id = faces_inds[:, 1, :, None] - faces_inds[:, 0, None, :]
            equal_id = torch.eq(equal_id, 0).sum(2).bool()
            replace = faces_inds[:, 1]
            replace[equal_id] = -1
            faces_inds = faces_inds.view(faces_inds.shape[0], -1)
            faces_inds, faces_inds_mask = ConformalMap.pad_faces(faces_inds)
        return faces_inds, faces_inds_mask

    @staticmethod
    def find_rolling_map(base, target):
        num_rings, num_faces, d = base.shape
        rolling_map = torch.arange(d, device=base.device).repeat(num_rings, num_faces, 1)
        for i in range(1, d):
            mask = base[:, :, i].eq(target[:, :, 0]) * base[:, :, (i + 1) % d].eq(target[:, :, 1])
            rolling_map[mask] = (torch.arange(d, device=base.device) + i) % d
        return rolling_map

    # return face mapper from 2d face ind to 3d face ind
    def faces_2to3_map(self, ds: Decimate) -> TS:
        faces_2d = self.uv_a[1]
        if self.in_low_phase:
            faces_2d = self.create_low_faces(self.num_rings)
            mask = torch.zeros(faces_2d.shape[:-1], dtype=torch.bool)
            vs_map = self.vs_map
        else:
            cum_sum = self.cum_sum_rings + torch.arange(self.cum_sum_rings.shape[0])
            vs_map = self.vs_map[:, :faces_2d.shape[1]]
            faces_2d = faces_2d - cum_sum[:, None, None]
            mask = vs_map.lt(0)
            faces_2d[mask] = 0
        target_faces_inds, target_faces_mask = self.create_faces_rings(ds)
        num_rings, num_target_faces = target_faces_inds.shape
        pad = torch.zeros(num_target_faces, dtype=torch.bool, device=vs_map.device)
        pad[0] = True
        # num_rings X ring_length X 3
        target_faces = ds.faces[target_faces_inds]
        target_faces[~target_faces_mask] = target_faces.max() + 1
        base_faces = torch.gather(vs_map, 1, faces_2d.view(num_rings, -1)).view(self.num_rings, -1, 3)
        diff = base_faces[:, :, None, :, None] - target_faces[:, None, :, None, :]
        diff = diff.eq(0).sum((-1, -2)).eq(3)
        diff[mask] = pad
        faces_map = torch.where(diff)
        faces_map = target_faces_inds[(faces_map[0], faces_map[2])].view(*base_faces.shape[:-1])
        faces_map_vs_inds = ds.faces[faces_map]
        faces_map_vs_inds[mask] = base_faces[mask] = -1
        rolling_map = self.find_rolling_map(base_faces, faces_map_vs_inds)
        faces_map[~target_faces_mask] = -1
        return faces_map, rolling_map

    @staticmethod
    def create_uv_a(mesh: T_Mesh, boundary_indices: T) -> T:
        num_boundary = boundary_indices.shape[0]
        boundary_coordinates = torch.zeros(num_boundary, 2)
        boundary_coordinates[:, 0] = torch.arange(num_boundary).float() * 2
        boundary_coordinates[torch.arange(num_boundary // 2) * 2, 0] += 1
        uv_a = mesh_utils.lscm(mesh, boundary_indices, boundary_coordinates)
        return uv_a

    @staticmethod
    def create_low_faces(num_rings: int):
        faces = torch.tensor([[0, 1, 2],
                              [0, 2, 3],
                              [0, 3, 1]],dtype=torch.int64)
        faces = faces.unsqueeze(0).repeat((num_rings, 1, 1))
        return faces

    def create_map_for_low(self, ds: Decimate) -> TS:
        num_rings, num_vs = ds.queue.shape[0], 4
        vs_map = torch.zeros(num_rings, num_vs, dtype=torch.int64)
        vs_map[:, 0] = ds.queue
        edge_a = ds.v2e[ds.queue, 0]
        edge_b = ds.skip_edge(edge_a)
        low_edges_id = torch.stack((edge_a, edge_b,  ds.skip_edge(edge_b)), dim=1)
        vs_map[:, 1:] = ds.edges[low_edges_id, 1]
        faces_a = self.create_low_faces(num_rings)
        faces_b = torch.tensor([[0, 1, 2]],dtype=torch.int64)
        cum_sum_a = torch.arange(num_rings) * 4
        self.cum_sum_rings = torch.arange(num_rings) * 3
        faces_b = faces_b.unsqueeze(0).repeat((num_rings, 1, 1))
        faces_a += cum_sum_a[:, None, None]
        faces_b += self.cum_sum_rings[:, None, None]
        faces_a, faces_b = faces_a.view(-1, 3), faces_b.view(-1, 3)
        boundary_indices_a = (cum_sum_a + 1).unsqueeze(1)
        boundary_indices_a = torch.cat((boundary_indices_a, boundary_indices_a + 1), dim=1).flatten()
        self.vs_a = vs_map.clone().flatten()
        self.vs_map = vs_map.clone()
        uv_a = self.create_uv_a((ds.vs[self.vs_a], faces_a), boundary_indices_a)
        uv_b_select = torch.arange(uv_a.shape[0])
        uv_b_select = (uv_b_select % 4).ne(0)
        uv_b = uv_a[uv_b_select]
        uv_a = uv_a, faces_a
        self.uv_b = uv_b, faces_b
        return uv_a

    def create_separated_rings(self, ds: Decimate) -> TS:
        edge_keys = torch.stack((ds.queue, ds.twins[ds.queue]), dim=1)
        num_rings, num_vs = ds.queue.shape[0], ds.max_degree * 2
        # num_conformals = min(2, num_rings)
        vs_map = torch.zeros(num_rings, num_vs, dtype=torch.int64) - 1  # map old vs id to vs id in uv
        faces = torch.zeros(num_rings, num_vs, 3, dtype=torch.int64)
        vs_map[:, :2] = ds.edges[ds.queue]
        degrees = torch.zeros(num_rings, dtype=torch.int64) + 2
        mask = torch.zeros(num_rings, num_vs, dtype=torch.bool)
        mask[:, :2] = True
        for j in range(2):
            faces[:, j, 0] = j
            faces[:, j, 1] = 1 - j
            faces[:, j, 2] = degrees
            start_edge = cur_edge = edge_keys[:, j]
            relevant_rings = torch.ones(num_rings, dtype=torch.bool)
            cur_face = degrees.unsqueeze(1).repeat(1, 3)
            cur_face[:, 0] = j
            cur_face[:, 2] = degrees + 1
            for i in range(ds.max_degree - 1):
                cur_edge = ds.skip_edge(cur_edge)
                relevant_rings = torch.ne(ds.skip_edge(cur_edge), start_edge) * relevant_rings
                vs_map[relevant_rings, degrees[relevant_rings]] = ds.edges[cur_edge][:, 1][relevant_rings]
                if i != 0:
                    faces[relevant_rings, degrees[relevant_rings] - 1] = cur_face[relevant_rings]
                    cur_face[relevant_rings, 1:] += 1
                mask[relevant_rings, degrees[relevant_rings]] = True
                degrees[relevant_rings] += 1
                if relevant_rings.sum() == 0:
                    break
            if j == 1:
                cur_face[:, 2] = 2
            faces[torch.arange(num_rings), degrees - 1] = cur_face
        vs = ds.vs[vs_map]
        cum_sum_a = self.cum_sum_with_zero(degrees)
        merged = self.create_merged_rings(vs, faces, mask, cum_sum_a, ds.updated_positions)
        vs_b, faces_b, boundary_indices_b, boundary_select, self.cum_sum_rings = merged
        faces += cum_sum_a[:, None, None]
        boundary_indices_a = (cum_sum_a + 2).unsqueeze(1)
        boundary_indices_a = torch.cat((boundary_indices_a, boundary_indices_a + 1), dim=1).flatten()
        self.vs_a = vs_map[mask].clone()
        self.vs_map = vs_map.clone()
        return faces[mask], boundary_indices_a, vs_b, faces_b, boundary_indices_b, boundary_select

    def create_map(self, ds: Decimate) -> Tuple[Union[T, T_Mesh], ...]:
        if self.in_low_phase:
            return self.create_map_for_low(ds)
        faces_a, boundary_indices_a, vs_b, faces_b, boundary_indices_b, boundary_s = self.create_separated_rings(ds)
        uv_a = self.create_uv_a((ds.vs[self.vs_a], faces_a), boundary_indices_a)
        uv_b = mesh_utils.lscm((vs_b, faces_b), boundary_indices_b, uv_a[boundary_s])
        uv_a = uv_a, faces_a
        self.uv_b = torch.index_select(uv_b, 1, torch.tensor([1, 0], dtype=torch.int64)), faces_b.clone()
        return uv_a

    @property
    def in_low_phase(self) -> bool:
        return self.phase is Decimate.DecimatePhase.LOW

    @property
    def num_rings(self) -> int:
        return self.cum_sum_rings.shape[0]

    def __init__(self, ds: Decimate):
        assert ds.decimate_available
        self.phase = ds.phase
        self.target_mesh = ds.get_mesh()
        self.vs_map: TN = None
        self.cum_sum_rings: TN = None
        self.uv_b: TNS = None
        self.vs_a: TN = None
        self.uv_a_raw = self.create_map(ds)
        self.uv_a = self.separate_uv(self.uv_a_raw)
        self.back_face_mapper = self.faces_2to3_map(ds)
        self.faces_map = self.create_faces_map(ds)
        self.reduced_mesh: Union[N, T_Mesh] = None
        self.back_mapper: TN = None
        self.back_mapper_vs: TN = None


class ConformalManger:

    def __getitem__(self, item: int) -> ConformalMap:
        return self.conformal_maps[item]

    @property
    def mesh(self) -> T_Mesh:
        return self.ds.mesh

    def decimate(self) -> ConformalManger:
        self.conformal_maps.append(ConformalMap(self.ds))
        self.ds, _, mapper = self.ds.decimate_all()
        self.conformal_maps[-1].complete_map(self.ds.get_mesh(), *mapper)
        return self

    def successive_decimate(self, num_trials=10) -> ConformalManger:
        if self.ds.min_vs == -1:
            return self.decimate()
        while len(self.ds) > self.ds.min_vs:
            decimate_happened = False
            for i in range(num_trials):
                if self.ds.decimate_available:
                    self.decimate()
                    decimate_happened = True
                    break
                else:
                    self.ds.reset_queue()
            if not decimate_happened:
                break
        return self

    def __call__(self, faces_inds: T, weights: T) -> T:
        for i in range(len(self.conformal_maps)):
            faces_inds, weights = self.conformal_maps[-i - 1](faces_inds, weights)
        if len(self.conformal_maps) > 0:
            mesh = self.conformal_maps[0].target_mesh
        else:
            mesh = self.mesh
        return mesh_utils.interpulate_vs(mesh, faces_inds, weights)

    def __init__(self, raw_mesh: T_Mesh, min_vs=-1):
        self.ds: Decimate = Decimate(raw_mesh, min_vs=min_vs)
        self.conformal_maps: List[ConformalMap] = []
