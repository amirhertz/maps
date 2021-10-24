from __future__ import annotations
from custom_types import *
from core.mesh_ds import MeshDS
from utils import mesh_utils


class SubdivideMesh:

    @staticmethod
    def get_base_args(ds: MeshDS) -> TS:
        base_faces_inds = ds.v2f[:, 0]
        base_weights = torch.zeros(len(ds), 3, device=ds.device)
        base_faces = ds.faces[base_faces_inds]
        find_vs_ind = (base_faces - torch.arange(len(ds), device=ds.device)[:, None]).eq(0)
        base_weights[find_vs_ind] = 1 #- 2 * EPSILON
        # base_weights[~find_vs_ind] = EPSILON
        return base_faces_inds, base_weights

    @staticmethod
    def get_up_args(ds: MeshDS) -> TS:
        faces_inds = ds.e2f[ds.unique_edges_id]
        faces = ds.faces[faces_inds]  # E X 3
        find_vs_ind = (faces[:, :, None] - ds.unique_edges[:, None, :].to(ds.device)).eq(0)  # E X 3 X 2
        find_vs_ind = find_vs_ind.sum(2).eq(1)
        weights = (find_vs_ind.float()) * .5
        # weights[~find_vs_ind] = EPSILON
        return faces_inds, weights

    def __init__(self, ds: Union[T_Mesh, MeshDS]):
        if not isinstance(ds, MeshDS):
            ds = MeshDS(ds)
        vs_t, faces_t = ds.get_mesh()
        device = ds.device
        base_inds = 3 * torch.arange(faces_t.shape[0], device=device).unsqueeze(1) + vs_t.shape[0]
        raw_edges = ds.edges.clone()
        mask = torch.zeros(raw_edges.shape[0], dtype=torch.bool, device=device)
        mask[ds.unique_edges_id] = 1
        mapper = torch.zeros(raw_edges.shape[0], dtype=torch.int64, device=device)
        mapper[ds.unique_edges_id] = torch.arange(ds.unique_edges_id.shape[0], device=device)
        mapper[ds.twins[ds.unique_edges_id]] = mapper[ds.unique_edges_id]
        faces_mid = [(torch.arange(3 * faces_t.shape[0], device=device) + vs_t.shape[0]).view(-1, 3)]
        faces_sr = [torch.cat([faces_t[:, i].unsqueeze(1), base_inds + i, base_inds + (i + 2) % 3], dim=1) for i in
                    range(3)]
        mapper = torch.cat((torch.arange(vs_t.shape[0], device=device), mapper + vs_t.shape[0])).to(device=vs_t.device)
        self.down_vs = ds.vs
        self.down_faces = faces_t
        up_faces = torch.stack(faces_sr + faces_mid, dim=1).view(-1, 3)
        self.up_faces = mapper[up_faces]
        self.down_len = vs_t.shape[0]
        self.vertices_select = raw_edges[mask]
        f_a, w_a = self.get_base_args(ds)
        f_b, w_b = self.get_up_args(ds)
        self.faces_inds = torch.cat((f_a, f_b), dim=0)
        self.weights = torch.cat((w_a, w_b), dim=0)

    def get_weights(self) -> TS:
        return self.faces_inds.clone(), self.weights.clone()

    @property
    def low_mesh(self) -> T_Mesh:
        return self.down_vs, self.down_faces

    @property
    def high_mesh(self) -> T_Mesh:
        vs = mesh_utils.interpulate_vs(self.low_mesh, *self.get_weights())
        return vs, self.up_faces

    def up(self, what: T) -> T:
        assert what.shape[0] == self.down_len
        mid = what[self.vertices_select]
        mid = (mid[:, 0] + mid[:, 1]) / 2  # faster than .mean(1)
        return torch.cat((what, mid), dim=0)

    def down(self, what):
        return what[: self.down_len]

    def down_sample(self, mesh: T_Mesh) -> T_Mesh:
        vs, faces = mesh
        return self.down(vs), self.down_faces

    def to(self, device: D):
        self.up_faces = self.up_faces.to(device, )
        self.down_faces = self.down_faces.to(device, )
        self.vertices_select = self.vertices_select.to(device, )
        return self

    weights_cache: Dict[int, T] = {}

    @staticmethod
    def get_level_weights(level: int):
        if level not in SubdivideMesh.weights_cache:
            if level == 0:
                weights = torch.eye(3, dtype=torch.float32)
            elif level == 1:
                weights = torch.tensor([[.5, .5, 0.],
                                        [0., .5, .5],
                                        [.5, 0., .5]])
            elif level == 2:
                weights = torch.tensor([[.5, .5, 0.],
                                        [0., .5, .5],
                                        [.5, 0., .5]])

            else:
                raise NotImplemented
            SubdivideMesh.weights_cache[level] = weights

        return SubdivideMesh.weights_cache[level]

    @staticmethod
    def compose_weights(low: SubdivideMesh, high: SubdivideMesh) -> TS:
        faces_inds_high, weights_high = high.get_weights()
        faces_inds_low, weights_ind_low = low.get_weights()
        fixed_inds = faces_inds_low.shape[0]
        translate_faces_inds, translate_weights = faces_inds_high[fixed_inds:], weights_high[fixed_inds:]
        translated_location = mesh_utils.interpulate_vs(low.high_mesh, translate_faces_inds, translate_weights)
        translate_faces_in_lower_inds = translate_faces_inds // 4
        translate_faces_in_lower = low.down_faces[translate_faces_in_lower_inds]
        weights_in_lower = mesh_utils.find_barycentric(translated_location, low.low_mesh[0][translate_faces_in_lower])
        faces_inds = torch.cat((faces_inds_low, translate_faces_in_lower_inds))
        weights = torch.cat((weights_ind_low, weights_in_lower))
        return faces_inds, weights

    def __call__(self, mesh: T_Mesh, features: TN = None) -> T_Mesh:
        vs, _ = mesh
        up_vs = self.up(vs)
        return up_vs, self.up_faces


if __name__ == '__main__':
    from utils import files_utils
    name = 'target'
    mesh = files_utils.load_mesh(f'low_{name}')
    mesh = mesh_utils.to_unit_edge(mesh)
    up = SubdivideMesh(mesh)
    mesh_mid = up(mesh)
    upper = SubdivideMesh(mesh_mid)
    vs = mesh_utils.interpulate_vs(mesh, *up.compose_weights(up, upper))
    # vs = interpulate_vs(mesh, up.faces_inds, up.weights)
    # files_utils.export_mesh(up_mesh, 'mesh_mid')
    files_utils.export_mesh(upper(mesh_mid), '{name}_mid')
    files_utils.export_mesh((vs, upper.up_faces), '{name}_mid_vs')


