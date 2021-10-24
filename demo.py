from core import conformal_manger
from utils import files_utils
import constants
from core import subdivide


def main():
    name = 'bunny'
    path = f'{constants.RAW_MESHES}{name}.obj'
    mesh = files_utils.load_mesh(path)
    num_faces = mesh[1].shape[0]
    cm = conformal_manger.ConformalManger(mesh, min_vs=num_faces // 32 + 2)
    cm.successive_decimate()
    sd = subdivide.SubdivideMesh(cm.ds)
    upper = subdivide.SubdivideMesh(sd.high_mesh)
    print(f'faces or: {num_faces} faces new: {cm.mesh[1].shape[0] * 16}')
    files_utils.export_mesh(cm.mesh, f'{constants.OUT}{name}_low.obj')
    samples = cm(*sd.compose_weights(sd, upper))
    files_utils.export_mesh((samples, upper.up_faces), f'{constants.OUT}{name}_high.obj')
    files_utils.export_mesh(upper.high_mesh, f'{constants.OUT}{name}_low_high')


if __name__ == '__main__':
    main()

