from __future__ import annotations
from custom_types import *
from utils import mesh_utils
from functools import reduce
from core.mesh_ds import MeshDS
from enum import Enum, auto
from utils import files_utils
import constants


class Decimate(MeshDS):

    class CollapsedType(Enum):

        INVALID = auto()
        REG = auto()
        LOW = auto()

    class DecimatePhase(Enum):

        REG = auto()
        LOW = auto()

    @staticmethod
    def create_merged_ring(vs: T, faces: T, updated_position: T) -> TS:
        vs, faces = vs.clone(), faces.clone()
        vs[1] = updated_position
        vs = vs[1:]
        faces = faces[2:] - 1
        faces[torch.eq(faces, -1)] = 0
        return vs, faces

    def create_separated_ring(self, edge_key, updated_position):
        edge_keys = torch.stack((edge_key, self.twins[edge_key]), dim=0)
        num_vs = self.max_degree * 2
        vs_map = torch.zeros(num_vs, dtype=torch.int64) - 1
        faces = torch.zeros(num_vs, 3, dtype=torch.int64)
        vs_map[:2] = self.edges[edge_key]
        degree = 2
        select_inds = torch.zeros(2, self.max_degree, dtype=torch.int64) - 1
        for j in range(2):
            start_deg = degree
            select_inds[j, 1] = 1 - j
            faces[j, 0] = j
            faces[j, 1] = 1 - j
            faces[j, 2] = degree
            # select_inds[j, 0] = degree
            start_edge = cur_edge = edge_keys[j]
            cur_face = torch.ones(3, dtype=torch.int64) * degree
            cur_face[0] = j
            cur_face[2] = degree + 1
            for i in range(self.max_degree - 1):
                cur_edge = self.skip_edge(cur_edge)
                if torch.eq(self.skip_edge(cur_edge), start_edge):
                    break
                vs_map[degree] = self.edges[cur_edge][1]
                select_inds[j, degree - start_deg + 2] = degree
                if i != 0:
                    faces[degree - 1] = cur_face
                    cur_face[1:] += 1
                degree += 1
            if j == 1:
                cur_face[2] = 2
            faces[degree - 1] = cur_face
        select_inds[0, 0] = select_inds[1, 2]
        select_inds[1, 0] = select_inds[0, 2]
        vs_map, faces_a = vs_map[: degree], faces[: degree]
        vs_a = self.vs[vs_map]
        vs_b, faces_b = self.create_merged_ring(vs_a, faces_a, updated_position)
        return vs_a, faces_a, vs_b, faces_b, [select_inds[0], select_inds[1]]

    def create_uv_maps(self, edge_key, updated_position) -> Tuple[Union[T, T_Mesh, TS], ...]:
        vs_a, faces_a, vs_b, faces_b, select_inds = self.create_separated_ring(edge_key, updated_position)
        boundary_coordinates = torch.zeros(2, 2)
        boundary_coordinates[1, 0] = 1.
        boundary_s = torch.arange(vs_a.shape[0] - 2) + 2
        uv_a = mesh_utils.lscm((vs_a, faces_a), torch.arange(2) + 2, boundary_coordinates)
        uv_b = mesh_utils.lscm((vs_b, faces_b), boundary_s - 1, uv_a[boundary_s])
        uv_a = uv_a, faces_a
        uv_b = torch.index_select(uv_b, 1, torch.tensor([1, 0], dtype=torch.int64)), faces_b.clone()
        select_inds = [inds[inds.ne(-1)] for inds in select_inds] + [torch.arange(vs_b.shape[0] - 1) + 1]
        return uv_a, uv_b, select_inds

    def check_skinny_faces(self, mesh: Union[T, T_Mesh], new_areas: TN = None) -> bool:
        if new_areas is None:
            new_areas, _ = mesh_utils.compute_face_areas(mesh)
        if type(mesh) is T:
            edges = mesh
        else:
            vs, faces = mesh
            edges = vs[faces]
        edges = reduce(
                       lambda a, b: a + b,
                       map(
                           lambda i: ((edges[:, i] - edges[:, (i + 1) % 3]) ** 2).sum(1),
                           range(3)
                       )
                )
        skinny_value = np.sqrt(48) * new_areas / edges
        return torch.ge(skinny_value, self.th_skinny).all().item()

    def skip_edge(self, edge_key):
        return self.twins[self.next[self.next[edge_key]]]

    def in_low_deg(self, edge_key) -> T:
        twins = self.twins[edge_key]
        skip_a = self.skip_edge(self.skip_edge(self.skip_edge(edge_key)))
        skip_b = self.skip_edge(self.skip_edge(self.skip_edge(twins)))
        return torch.eq(skip_a, edge_key) + torch.eq(skip_b, twins)

    def edges_are_valid(self, edges_keys):
        return self.in_low_deg(edges_keys) +\
               ~(self.in_low_deg(self.next[edges_keys]) + self.in_low_deg(self.next[self.twins[edges_keys]]))

    def decimate_invalids(self) -> Tuple[Decimate, T, TS]:
        if not self.decimate_available:
            self.reset_queue()
            return self, self.merged_vs, (mesh_utils.create_mapper(self.vs_mask), self.faces_mask)
        removed_vs = []
        edge_a = self.v2e[self.queue, 0]
        edge_b = self.skip_edge(edge_a)
        low_edges_id = torch.stack((edge_a, edge_b,  self.skip_edge(edge_b)), dim=1)
        faces_to_update_id = self.e2f[low_edges_id[:, -1]]
        faces_to_remove = self.e2f[low_edges_id[:, :2]]
        where_to_remove = torch.where(torch.eq(self.faces[faces_to_update_id], self.queue[:, None]))[1]
        self.vs_mask[self.queue] = False
        self.faces_mask[faces_to_remove.flatten()] = False
        vs_to_merge = self.edges[low_edges_id[:, 1], 1]
        self.faces[faces_to_update_id, where_to_remove] = vs_to_merge
        removed_vs.append(self.queue)
        return self.clean()

    def decimate_all(self) -> Tuple[Decimate, T, TS]:
        if self.in_low_phase:
            return self.decimate_invalids()
        edge_keys = torch.stack((self.queue, self.twins[self.queue]), dim=1)
        removed_face_id = self.e2f[edge_keys]
        removed_edge_key_a = self.next[edge_keys]
        removed_edge_key_b = self.next[removed_edge_key_a]
        merged_vs = self.edges[self.queue]
        # update masks
        self.faces_mask[removed_face_id.flatten()] = False
        self.vs_mask[merged_vs[:, 1]] = False
        # update vs position
        self.vs[merged_vs[:, 0]] = self.updated_positions
        # update edges ds
        self.twins[self.twins[removed_edge_key_b]] = self.twins[removed_edge_key_a]
        self.twins[self.twins[removed_edge_key_a]] = self.twins[removed_edge_key_b]
        # update faces vs key
        to_update_faces_inds = self.v2f[merged_vs[:, 1]]
        to_update_faces = self.faces[to_update_faces_inds]
        faces_mask = self.ring_mask[merged_vs[:, 1]]
        update_vs_key = merged_vs[:, 0].unsqueeze(1).expand_as(to_update_faces_inds)
        to_update_inds = torch.eq(to_update_faces, merged_vs[:, 1][:, None, None])
        to_update_faces_inds = to_update_faces_inds[faces_mask]
        to_update_inds = torch.where(to_update_inds[faces_mask])[1]
        self.faces[to_update_faces_inds, to_update_inds] = update_vs_key[faces_mask]
        self.merged_vs = (merged_vs[:, 0], self.merged_vs)
        self.removed_vs = (merged_vs[:, 1], self.removed_vs)
        return self.clean()

    def __call__(self) -> Tuple[Decimate, T, TS]:
        return self.decimate_all()

    def check_uv(self, edge_key, updated_position) -> bool:
        uv_a, uv_b, select = self.create_uv_maps(edge_key, updated_position)
        return \
            self.check_skinny_faces(uv_b) and \
            mesh_utils.check_sign_area(uv_a, uv_b) and \
            mesh_utils.check_circle_angles(uv_a, 0, select[0]) and \
            mesh_utils.check_circle_angles(uv_a, 1, select[1]) and \
            mesh_utils.check_circle_angles(uv_b, 0, select[2])

    def check_vs_ring(self, edge_keys: T, updated_position: T) -> T:
        edge_vs = self.edges[edge_keys]  # e 2
        if edge_vs.dim() == 1:
            edge_vs = edge_vs.unsqueeze(0)
        vs_ring = self.edges[self.v2e[edge_vs]][:, :, :, 1]  # e 2 r
        diff_rings = (vs_ring[:, 0, :, None] - vs_ring[:, 1, None, :]).abs().min(2)[0]
        diff_rings[~self.ring_mask[edge_vs[:, 0]]] = 1
        shared_vs = torch.eq(diff_rings, 0).sum(1)
        return torch.eq(shared_vs, 2) and self.check_uv(edge_keys, updated_position)

    def check_single_flip(self, edge_key: int, new_position: T) -> CollapsedType:
        edge_keys = torch.tensor([edge_key, self.twins[edge_key]], dtype=torch.int64)
        removed_face_id = self.e2f[edge_keys]
        merged_vs = self.edges[edge_key]
        faces_mask_a = self.ring_mask[merged_vs[0]], self.ring_mask[merged_vs[1]]
        to_update_faces_inds = torch.cat((self.v2f[merged_vs[0]][faces_mask_a[0]],
                                         self.v2f[merged_vs[1]][faces_mask_a[1]]), 0).unique()
        faces_mask_b = torch.eq(to_update_faces_inds[:, None], removed_face_id[None, :]).sum(1).bool()
        to_update_faces_inds = to_update_faces_inds[~faces_mask_b]
        to_update_faces = self.faces[to_update_faces_inds]
        to_update_inds = torch.eq(to_update_faces[:, :, None], merged_vs[None, None, :]).sum(2).bool()
        to_update_inds = torch.where(to_update_inds)
        vs_faces = self.vs[to_update_faces].clone()
        vs_faces[to_update_inds] = new_position
        new_areas, new_normals = mesh_utils.compute_face_areas(vs_faces)
        skinny_check = self.check_skinny_faces(vs_faces, new_areas)  # and new_areas.min() < EPSILON ** 2
        if skinny_check:
            delta_n = torch.einsum('nd,nd->n', [new_normals, self.face_normals[to_update_faces_inds]])
            delta_n = torch.ge(delta_n, self.th_flip)
            if delta_n.all().item():
                return self.CollapsedType.REG
        return self.CollapsedType.INVALID

    def get_decimated_edges(self) -> T:
        merged_edges = torch.cat([self.edges[self.queue]], dim=0)
        return merged_edges

    def compute_edge_quadrics(self, normals) -> T:
        point_on_face = self.vs[self.faces[:, 0]]
        d = -torch.einsum('fd,fd->f', [normals, point_on_face])
        normals = torch.cat((normals, d.unsqueeze(1)), dim=1)
        face_quadrics = normals[:, :, None] * normals[:, None, :]  # out product
        vs_quadrics = face_quadrics[self.v2f]
        vs_quadrics = vs_quadrics * self.ring_mask.float()[:, :, None, None]
        vs_quadrics = vs_quadrics.sum(1) / self.vs_degree[:, None, None].float()
        edge_quadrics = vs_quadrics[self.unique_edges].sum(1)
        return edge_quadrics

    @staticmethod
    def adjust_quadrics(edge_quadrics: T) -> T:
        quadrics = edge_quadrics.clone()
        quadrics[:, -1, :] = 0
        quadrics[:, -1, -1] = 1
        return quadrics

    def get_vs_after_decimation(self, edge_quadrics):
        try:
            vs_after_decimation = torch.inverse(edge_quadrics)[:, :, 3]
        except RuntimeError:
            vs_after_decimation = torch.zeros(edge_quadrics.shape[0], 4)
            _, v, _ = torch.svd(edge_quadrics)
            valid = torch.eq(torch.ne(v, 0).sum(1), 4)
            mid_points = self.vs[self.unique_edges[~valid]].mean(1)
            mid_points = torch.cat((mid_points, torch.ones(mid_points.shape[0], 1)), dim=1)
            vs_after_decimation[valid] = torch.inverse(edge_quadrics[valid])[:, :, 3]
            vs_after_decimation[~valid] = mid_points
        return vs_after_decimation

    def compute_edges_errors(self) -> TS:
        edge_quadrics = self.compute_edge_quadrics(self.face_normals)
        for_inverse = self.adjust_quadrics(edge_quadrics)
        vs_after_decimation = self.get_vs_after_decimation(for_inverse)
        error = torch.einsum('ed,eda,ea->e', [vs_after_decimation, edge_quadrics, vs_after_decimation])
        return error, vs_after_decimation[:, : -1]

    @property
    def flip_phase(self) -> DecimatePhase:
        if self.in_low_phase:
            return self.DecimatePhase.REG
        else:
            return self.DecimatePhase.LOW

    @property
    def in_low_phase(self) -> bool:
        return self.phase is self.DecimatePhase.LOW

    @property
    def vs_mask(self) -> T:
        return self.decimate_masks[0]

    @property
    def faces_mask(self) -> T:
        return self.decimate_masks[1]

    def collect_ugly_ring(self, edge_key):
        ugly_ring = []
        for i in range(3):
            ugly_ring.append(edge_key)
            ugly_ring.append(self.twins[edge_key])
            ugly_ring.append(self.next[edge_key])
            ugly_ring.append(self.next[ugly_ring[-1]])
            edge_key = self.next[self.twins[self.next[edge_key]]]
        return ugly_ring

    def get_edge_ring(self, edge_key: int, new_position: T) -> Tuple[CollapsedType, Union[T, int]]:
        cp_type = self.CollapsedType.REG
        all_edge = []
        if self.in_low_deg(self.next[edge_key]):
            all_edge += self.collect_ugly_ring(edge_key)
            cp_type = self.CollapsedType.LOW
        if self.in_low_deg(self.next[self.twins[edge_key]]):
            all_edge += self.collect_ugly_ring(self.twins[edge_key])
            cp_type = self.CollapsedType.LOW
        if cp_type is not self.CollapsedType.LOW:
            cp_type = self.check_single_flip(edge_key, new_position)
            if cp_type is self.CollapsedType.INVALID:
                return cp_type, -1
            all_edge = self.v2e[self.edges[edge_key]].flatten().unique()
            all_edge = all_edge[torch.ge(all_edge, 0)]
            all_edges_twins = self.twins[all_edge]
            next_all_edges = self.next[all_edge]
            next_all_edges_twins = self.twins[next_all_edges]
            all_edge = torch.cat((all_edge, all_edges_twins, next_all_edges, next_all_edges_twins), dim=0)
            next_all_edges_twins = self.next[next_all_edges_twins]
            for edge_id in next_all_edges_twins:
                if edge_id not in next_all_edges_twins:
                    print("err")
        else:
            all_edge = torch.tensor(all_edge, dtype=torch.int64)
        all_edge = all_edge.unique()
        all_edge = self.v2e[self.edges[all_edge]].flatten().unique()
        all_edge = all_edge[torch.ge(all_edge, 0)]
        all_edges_twins = self.twins[all_edge]
        all_edge = torch.cat((all_edge, all_edges_twins), dim=0)
        return cp_type, all_edge.unique()

    def check_edge(self, edge_id: int) -> bool:
        return True

    def extract_low_independent_set(self) -> TS:
        queue = torch.where(self.vs_degree.eq(3))[0]
        if queue.shape[0] > 0:
            vs_mid = self.vs[queue]
            edge_a = self.v2e[queue, 0]
            edge_b = self.skip_edge(edge_a)
            edge_c = self.skip_edge(edge_b)
            edge_d = self.skip_edge(edge_c)
            assert (edge_a - edge_d).eq(0).all().item()
            triangle = torch.stack((edge_a, edge_b, edge_c), dim=1)
            triangle = self.edges[triangle][:, :, 1]
            triangle = self.vs[triangle]
            is_valid = mesh_utils.vs_over_triangle(vs_mid, triangle)
            queue = queue[is_valid]
        return queue, torch.zeros(0)

    def extract_independent_set(self) -> TS:
        if self.in_low_phase:
            return self.extract_low_independent_set()
        errors, updated_positions = self.compute_edges_errors()
        independent_mask = torch.ones_like(self.twins, dtype=torch.bool)
        # edge_order = torch.argsort(errors, dim=0)
        # to_rand = int(queue.shape[0] * .2)
        # queue[: to_rand] = queue[torch.argsort(torch.rand(to_rand), dim=0)]
        edge_order = torch.argsort(torch.rand(errors.shape), dim=0)
        # edge_order = torch.arange(errors.shape[0])
        queue = self.unique_edges_id[edge_order]
        updated_positions = updated_positions[edge_order]
        for i in range(queue.shape[0]):
            if independent_mask[queue[i]]:
                if self.check_edge(queue[i]):
                    cp_type, edge_ring = self.get_edge_ring(queue[i], updated_positions[i])
                    if cp_type is self.CollapsedType.INVALID or cp_type is self.CollapsedType.LOW:
                        independent_mask[queue[i]] = False
                    elif self.check_vs_ring(queue[i], updated_positions[i]):
                        independent_mask[edge_ring] = False
                        independent_mask[queue[i]] = True
                    else:
                        independent_mask[queue[i]] = False
                else:
                    independent_mask[queue[i]] = False
        independent_mask = independent_mask[queue]
        queue = queue[independent_mask]
        valid = self.edges_are_valid(queue)
        assert valid.all().item()
        updated_positions = updated_positions[independent_mask][valid]
        if self.min_vs > 0 and len(self) - len(queue) < self.min_vs:
            queue = queue[: max(len(self) - self.min_vs, 0)]
            updated_positions = updated_positions[: len(queue)]
        return queue,  updated_positions

    def constructor(self, mesh: T_Mesh, vs_mapper: T) -> Decimate:
        return Decimate(mesh, self.flip_phase, self.min_vs)

    def clean(self) -> Tuple[Decimate, T, TS]:
        vs = self.vs[self.vs_mask]
        faces = self.faces[self.faces_mask]
        mapper = mesh_utils.create_mapper(self.vs_mask)
        faces = mapper[faces]
        return self.constructor((vs, faces), mapper), self.merged_vs, (mapper, self.faces_mask)

    @property
    def decimate_available(self) -> bool:
        return self.queue.shape[0] > 0

    def __len__(self):
        return self.vs.shape[0]

    def reset_queue(self):
        self.phase = self.flip_phase
        self.queue, self.updated_positions = self.extract_independent_set()

    def __init__(self, raw_mesh: T_Mesh, phase: DecimatePhase = DecimatePhase.LOW, min_vs: int = -1):
        self.merged_vs = self.removed_vs = None
        super(Decimate, self).__init__(raw_mesh)
        self.phase = phase
        self.th_flip = .2
        self.th_skinny = .1
        self.min_vs = min_vs
        self.decimate_masks = tuple(
            map(lambda item: torch.ones(item.shape[0], dtype=torch.bool), [self.vs, self.faces]))
        self.face_areas, self.face_normals = mesh_utils.compute_face_areas(raw_mesh)
        self.queue, self.updated_positions = self.extract_independent_set()
        if self.phase is self.DecimatePhase.LOW and not self.decimate_available:
            self.reset_queue()


def decimate_mesh(mesh: T_Mesh, min_vs: int) -> T_Mesh:
    max_trial = 8
    un_decimate_counter = 0
    mesh_: T_Mesh = tuple(mesh_utils.clone(*mesh))
    ds = Decimate(mesh_)
    prev_len = len(ds)
    for i in range(100):
        ds = ds.decimate_all()[0]
        un_decimate_counter = (un_decimate_counter + int(prev_len == len(ds))) * int(prev_len == len(ds))
        prev_len = len(ds)
        print(len(ds))
        if len(ds) <= min_vs or un_decimate_counter > max_trial:
            break
    return ds.mesh


def main():
    name = 'bunny'
    min_vs = 300
    path = f'{constants.RAW_MESHES}{name}.obj'
    mesh = files_utils.load_mesh(path)
    mesh_d = decimate_mesh(mesh, min_vs)
    files_utils.export_mesh(mesh_d, f"{constants.OUT}{name}_300")


if __name__ == '__main__':
    main()
