from custom_types import *
from constants import EPSILON
import igl


def scale_all(*values: T):
    max_val = max([val.max().item() for val in values])
    min_val = min([val.min().item() for val in values])
    scale = max_val - min_val
    values = [(val - min_val) / scale for val in values]
    if len(values) == 1:
        return values[0]
    return values


def get_faces_normals(mesh: Union[T_Mesh, T]) -> T:
    if type(mesh) is not T:
        vs, faces = mesh
        vs_faces = vs[faces]
    else:
        vs_faces = mesh
    if vs_faces.shape[-1] == 2:
        vs_faces = torch.cat(
            (vs_faces, torch.zeros(*vs_faces.shape[:2], 1, dtype=vs_faces.dtype, device=vs_faces.device)), dim=2)
    face_normals = torch.cross(vs_faces[:, 1, :] - vs_faces[:, 0, :], vs_faces[:, 2, :] - vs_faces[:, 1, :])
    return face_normals


def compute_face_areas(mesh: Union[T_Mesh, T]) -> TS:
    face_normals = get_faces_normals(mesh)
    face_areas = torch.norm(face_normals, p=2, dim=1)
    face_areas_ = face_areas.clone()
    face_areas_[torch.eq(face_areas_, 0)] = 1
    face_normals = face_normals / face_areas_[:, None]
    face_areas = 0.5 * face_areas
    return face_areas, face_normals


def check_sign_area(*meshes: T_Mesh) -> bool:
    for mesh in meshes:
        face_normals = get_faces_normals(mesh)
        if not face_normals[:, 2].gt(0).all():
            return False
    return True


def to_numpy(*tensors: T) -> ARRAYS:
    params = [param.detach().cpu().numpy() if type(param) is T else param for param in tensors]
    return params


def create_mapper(mask: T) -> T:
    mapper = torch.zeros(mask.shape[0], dtype=torch.int64, device=mask.device) - 1
    mapper[mask] = torch.arange(mask.sum().item(), device=mask.device)
    return mapper


def mesh_center(mesh: T_Mesh):
    return mesh[0].mean(0)


def to_center(vs):
    max_vals = vs.max(0)[0]
    min_vals = vs.min(0)[0]
    center = (max_vals + min_vals) / 2
    vs -= center[None, :]
    return vs


def to_unit_sphere(mesh: T_Mesh,  in_place: bool = True, scale=1.) -> T_Mesh:
    vs, faces = mesh
    if not in_place:
        vs = vs.clone()
    vs = to_center(vs)
    norm = vs.norm(2, dim=1).max()
    vs *= scale * norm ** -1
    return vs, faces


def scale_from_ref(mesh: T_Mesh, center: T, scale: float, in_place: bool = True) -> T_Mesh:
    vs, faces = mesh
    if not in_place:
        vs = vs.clone()
    vs -= center[None, :]
    vs *= scale
    return vs, faces


def to_unit_cube(*meshes: T_Mesh_T, scale=1, in_place: bool = True) -> Tuple[Union[T_Mesh_T, Tuple[T_Mesh_T, ...]], Tuple[T, float]]:
    remove_me = 0
    meshes = [(mesh, remove_me) if type(mesh) is T else mesh for mesh in meshes]
    vs, faces = meshes[0]
    max_vals = vs.max(0)[0]
    min_vals = vs.min(0)[0]
    max_range = (max_vals - min_vals).max() / 2
    center = (max_vals + min_vals) / 2
    meshes_ = []
    scale = float(scale / max_range)
    for mesh in meshes:
        vs_, faces_ = scale_from_ref(mesh, center, scale)
        meshes_.append(vs_ if faces_ is remove_me else (vs_, faces_))
    if len(meshes_) == 1:
        meshes_ = meshes_[0]
    return meshes_, (center, scale)


def get_edges_ind(mesh: T_Mesh) -> T:
    vs, faces = mesh
    raw_edges = torch.cat([faces[:, [i, (i + 1) % 3]] for i in range(3)]).sort()
    raw_edges = raw_edges[0].cpu().numpy()
    edges = {(int(edge[0]), int(edge[1])) for edge in raw_edges}
    edges = torch.tensor(list(edges), dtype=torch.int64, device=faces.device)
    return edges


def edge_lengths(mesh: T_Mesh, edges_ind: TN = None) -> T:
    vs, faces = mesh
    if edges_ind is None:
        edges_ind = get_edges_ind(mesh)
    edges = vs[edges_ind]
    return torch.norm(edges[:, 0] - edges[:, 1], 2, dim=1)


# in place
def to_unit_edge(*meshes: T_Mesh) -> Tuple[Union[T_Mesh, Tuple[T_Mesh, ...]], Tuple[T, float]]:
    ref = meshes[0]
    center = ref[0].mean(0)
    ratio = edge_lengths(ref).mean().item()
    for mesh in meshes:
        vs, _ = mesh
        vs -= center[None, :].to(vs.device)
        vs /= ratio
    if len(meshes) == 1:
        meshes = meshes[0]
    return meshes, (center, ratio)


def to(tensors, device: D) -> Union[T_Mesh, TS, T]:
    out = []
    for tensor in tensors:
        if type(tensor) is T:
            out.append(tensor.to(device, ))
        elif type(tensor) is tuple or type(tensors) is List:
            out.append(to(list(tensor), device))
        else:
            out.append(tensor)
    if len(tensors) == 1:
        return out[0]
    else:
        return tuple(out)


def clone(*tensors: Union[T, TS]) -> Union[TS, T_Mesh]:
    out = []
    for t in tensors:
        if type(t) is T:
            out.append(t.clone())
        else:
            out.append(clone(*t))
    return out


def get_box(w: float, h: float, d: float) -> T_Mesh:
    vs = [[0, 0, 0], [w, 0, 0], [0, d, 0], [w, d, 0],
          [0, 0, h], [w, 0, h], [0, d, h], [w, d, h]]
    faces = [[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6],
             [0, 1, 5], [0, 5, 4], [2, 6, 7], [3, 2, 7],
             [1, 3, 5], [3, 7, 5], [0, 4, 2], [2, 4, 6]]
    return torch.tensor(vs, dtype=torch.float32), torch.tensor(faces, dtype=torch.int64)


def normalize(t: T):
    t = t / t.norm(2, dim=1)[:, None]
    return t


def interpolate_vs(mesh: T_Mesh, faces_inds: T, weights: T) -> T:
    vs = mesh[0][mesh[1][faces_inds]]
    vs = vs * weights[:, :, None]
    return vs.sum(1)


def sample_uvw(shape, device: D):
    u, v = torch.rand(*shape, device=device), torch.rand(*shape, device=device)
    mask = (u + v).gt(1)
    u[mask], v[mask] = -u[mask] + 1, -v[mask] + 1
    w = -u - v + 1
    uvw = torch.stack([u, v, w], dim=len(shape))
    return uvw


def get_sampled_fe(fe: T, mesh: T_Mesh, face_ids: T, uvw: TN) -> T:
    # to_squeeze =
    if fe.dim() == 1:
        fe = fe.unsqueeze(1)
    if uvw is None:
        fe_iner = fe[face_ids]
    else:
        vs_ids = mesh[1][face_ids]
        fe_unrolled = fe[vs_ids]
        fe_iner = torch.einsum('sad,sa->sd', fe_unrolled, uvw)
    # if to_squeeze:
    #     fe_iner = fe_iner.squeeze_(1)
    return fe_iner


def sample_on_faces(mesh: T_Mesh,  num_samples: int) -> TS:
    vs, faces = mesh
    uvw = sample_uvw([faces.shape[0], num_samples], vs.device)
    samples = torch.einsum('fad,fna->fnd', vs[faces], uvw)
    return samples, uvw


class SampleBy(Enum):
    AREAS = 0
    FACES = 1
    HYB = 2


def sample_on_mesh(mesh: T_Mesh, num_samples: int, face_areas: TN = None,
                   sample_s: SampleBy = SampleBy.HYB) -> TNS:
    vs, faces = mesh
    if faces is None:  # sample from pc
        uvw = None
        if vs.shape[0] < num_samples:
            chosen_faces_inds = torch.arange(vs.shape[0])
        else:
            chosen_faces_inds = torch.argsort(torch.rand(vs.shape[0]))[:num_samples]
        samples = vs[chosen_faces_inds]
    else:
        weighted_p = []
        if sample_s == SampleBy.AREAS or sample_s == SampleBy.HYB:
            if face_areas is None:
                face_areas, _ = compute_face_areas(mesh)
            face_areas[torch.isnan(face_areas)] = 0
            weighted_p.append(face_areas / face_areas.sum())
        if sample_s == SampleBy.FACES or sample_s == SampleBy.HYB:
            weighted_p.append(torch.ones(mesh[1].shape[0], device=mesh[0].device))
        chosen_faces_inds = [torch.multinomial(weights, num_samples // len(weighted_p), replacement=True) for weights in weighted_p]
        if sample_s == SampleBy.HYB:
            chosen_faces_inds = torch.cat(chosen_faces_inds, dim=0)
        chosen_faces = faces[chosen_faces_inds]
        uvw = sample_uvw([num_samples], vs.device)
        samples = torch.einsum('sf,sfd->sd', uvw, vs[chosen_faces])
    return samples, chosen_faces_inds, uvw


def get_samples(mesh: T_Mesh, num_samples: int, sample_s: SampleBy, *features: T) -> Union[T, TS]:
    samples, face_ids, uvw = sample_on_mesh(mesh, num_samples, sample_s=sample_s)
    if len(features) > 0:
        samples = [samples] + [get_sampled_fe(fe, mesh, face_ids, uvw) for fe in features]
    return samples, face_ids, uvw


def find_barycentric(vs: T, triangles: T) -> T:

    def compute_barycentric(ind):
        triangles[:, ind] = vs
        alpha = compute_face_areas(triangles)[0] / areas
        triangles[:, ind] = recover[:, ind]
        return alpha

    device, dtype = vs.device, vs.dtype
    vs = vs.to(device, dtype=torch.float64)
    triangles = triangles.to(device, dtype=torch.float64)
    areas, _ = compute_face_areas(triangles)
    recover = triangles.clone()
    barycentric = [compute_barycentric(i) for i in range(3)]
    barycentric = torch.stack(barycentric, dim=1)
    # assert barycentric.sum(1).max().item() <= 1 + EPSILON
    return barycentric.to(device, dtype=dtype)


def from_barycentric(mesh: Union[T_Mesh, T], face_ids: T, weights: T) -> T:
    if type(mesh) is not T:
        triangles: T = mesh[0][mesh[1]]
    else:
        triangles: T = mesh
    to_squeeze = weights.dim() == 1
    if to_squeeze:
        weights = weights.unsqueeze(0)
        face_ids = face_ids.unsqueeze(0)
    vs = torch.einsum('nad,na->nd', triangles[face_ids], weights)
    if to_squeeze:
        vs = vs.squeeze(0)
    return vs


def check_circle_angles(mesh: T_Mesh, center_ind: int, select: T) -> bool:
    vs, _ = mesh
    all_vecs = vs[select] - vs[center_ind][None, :]
    all_vecs = all_vecs / all_vecs.norm(2, 1)[:, None]
    all_vecs = torch.cat([all_vecs, all_vecs[:1]], dim=0)
    all_cos = torch.einsum('nd,nd->n', all_vecs[1:], all_vecs[:-1])
    all_angles = torch.acos_(all_cos)
    all_angles = all_angles.sum()
    return (all_angles - 2 * np.pi).abs() < EPSILON


def vs_over_triangle(vs_mid: T, triangle: T, normals=None) -> T:
    if vs_mid.dim() == 1:
        vs_mid = vs_mid.unsqueeze(0)
        triangle = triangle.unsqueeze(0)
    if normals is None:
        _, normals = compute_face_areas(triangle)
    select = torch.arange(3)
    d_vs = vs_mid[:, None, :] - triangle
    d_f = triangle[:, select] - triangle[:, (select + 1) % 3]
    all_cross = torch.cross(d_vs, d_f, dim=2)
    all_dots = torch.einsum('nd,nad->na', normals, all_cross)
    is_over = all_dots.ge(0).long().sum(1).eq(3)
    return is_over


def igl_prepare(*dtypes):

    def decoder(func):

        def wrapper(*args, **kwargs):
            mesh = args[0]
            device, dtype = mesh[0].device, mesh[0].dtype
            vs, faces = to_numpy(*mesh)
            result = func((vs, faces), *args[1:], **kwargs)
            return to_torch(result, device)

        to_torch = to_torch_singe if len(dtypes) == 1 else to_torch_multi

        return wrapper

    def to_torch_singe(result, device):
        return torch.from_numpy(result).to(device, dtype=dtypes[0])

    def to_torch_multi(result, device):
        return [torch.from_numpy(r).to(device, dtype=dtype) for r, dtype in zip(result, dtypes)]

    return decoder


@igl_prepare(torch.float32, torch.int64)
def decimate_igl(mesh, num_faces: int):
    if mesh[1].shape[0] <= num_faces:
        return mesh
    vs, faces, _ = igl.remove_duplicates(*mesh, 1e-8)
    return igl.decimate(vs, faces, num_faces)[1:3]


@igl_prepare(torch.float32)
def gaussian_curvature(mesh: T_Mesh) -> T:
    gc = igl.gaussian_curvature(*mesh)
    return gc


@igl_prepare(torch.float32)
def per_vertex_normals_igl(mesh: T_Mesh, weighting: int = 0) -> T:
    normals = igl.per_vertex_normals(*mesh, weighting)
    return normals


@igl_prepare(torch.float32, torch.int64)
def remove_duplicate_vertices(mesh: T_Mesh, epsilon=1e-7) -> T_Mesh:
    vs, _, _, faces = igl.remove_duplicate_vertices(*mesh, epsilon)
    return vs, faces


@igl_prepare(torch.float32)
def winding_number_igl(mesh: T_Mesh, query: T) -> T:
    query = query.cpu().numpy()
    return igl.fast_winding_number_for_meshes(*mesh, query)


@igl_prepare(torch.float32, torch.float32, torch.float32, torch.float32)
def principal_curvature(mesh: T_Mesh) -> TS:
    out = igl.principal_curvature(*mesh)
    min_dir, max_dir, min_val, max_val = out
    return min_dir, max_dir, min_val, max_val


def get_inside_outside(points: T, mesh: T_Mesh) -> T:
    device = points.device
    points = points.numpy()
    vs, faces = mesh[0].numpy(), mesh[1].numpy()
    winding_numbers = igl.fast_winding_number_for_meshes(vs, faces, points)
    winding_numbers = torch.from_numpy(winding_numbers)
    inside_outside = winding_numbers.lt(.5).float() * 2 - 1
    return inside_outside.to(device)


@igl_prepare(torch.float32)
def lscm(mesh: T_Mesh, boundary_indices: T, boundary_coordinates: T) -> T:
    boundary_indices, boundary_coordinates = boundary_indices.numpy(), boundary_coordinates.numpy()
    check, uv = igl.lscm(*mesh, boundary_indices, boundary_coordinates)
    return uv


def interpulate_vs(mesh: T_Mesh, faces_inds: T, weights: T) -> T:
    vs = mesh[0][mesh[1][faces_inds]]
    vs = vs * weights[:, :, None]
    return vs.sum(1)
