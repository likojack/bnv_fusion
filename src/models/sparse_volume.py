import numpy as np
import open3d as o3d
import open3d.core as o3c
from skimage.measure import marching_cubes
import torch
import torch.nn.functional as F
import torch.utils.dlpack
import trimesh

from src.models.fusion.utils import get_neighbors
import src.utils.voxel_utils as voxel_utils
import src.utils.o3d_helper as o3d_helper


class SparseVolume:
    def __init__(self, n_feats, voxel_size, dimensions, min_pts_in_grid, capacity=100000, device="cuda:0") -> None:
        """
        Based on hash map implementation in Open3D.
        """
        min_coords, max_coords, n_xyz = voxel_utils.get_world_range(
            dimensions, voxel_size)
        self.device = device
        self.dimensions = dimensions
        self.voxel_size = voxel_size
        self.o3c_device = o3c.Device(device)
        self.min_coords = torch.from_numpy(min_coords).float().to(device)
        self.max_coords = torch.from_numpy(max_coords).float().to(device)
        self.n_xyz = torch.from_numpy(np.asarray(n_xyz)).long().to(device)
        self.n_feats = n_feats
        self.min_pts_in_grid = min_pts_in_grid
        self.reset(capacity)

        self.avg_n_pts = 0
        self.n_pts_list = []
        self.n_frames = 0
        self.min_pts = 1000
        self.max_pts = 0

    def track_n_pts(self, n_pts):
        self.n_pts_list.append(float(n_pts))
        self.avg_n_pts = (self.avg_n_pts * self.n_frames + n_pts) / (self.n_frames + 1)
        self.n_frames += 1
        self.min_pts = min(self.min_pts, n_pts)
        self.max_pts = max(self.max_pts, n_pts)

    def print_statistic(self):
        print("===========")
        p = np.percentile(self.n_pts_list, [25, 50, 75])
        self.per_25 = p[0]
        self.per_50 = p[1]
        self.per_75 = p[2]
        print(f"25%: {p[0]}, 50%: {p[1]}, 75%:{p[2]}")
        print(f"mean: {self.avg_n_pts}, min: {self.min_pts}, max:{self.max_pts}")
        print("===========")

    def to_tensor(self):
        """ store all active values to pytorch tensor
        """

        active_buf_indices = self.indexer.active_buf_indices().to(o3c.int64)
        capacity = len(active_buf_indices)
        self.tensor_indexer = o3c.HashMap(
            capacity,
            key_dtype=o3c.int64,
            key_element_shape=(3,),
            value_dtype=o3c.int64,
            value_element_shape=(1,),
            device=o3c.Device(self.device)
        )

        active_keys = self.indexer.key_tensor()[active_buf_indices].to(o3c.int64)
        features = self.indexer.value_tensor(0)[active_buf_indices]
        weights = self.indexer.value_tensor(1)[active_buf_indices]
        num_hits = self.indexer.value_tensor(2)[active_buf_indices]

        indexer_value = o3c.Tensor.from_dlpack(
            torch.utils.dlpack.to_dlpack(
                torch.arange(capacity, device=self.device)
            )
        )
        buf_indices, masks = self.tensor_indexer.insert(active_keys, indexer_value)
        masks = masks.cpu().numpy() if "cuda" in self.device else masks.numpy()
        assert masks.all()

        self.active_coordinates = torch.utils.dlpack.from_dlpack(active_keys.to_dlpack())
        self.features = torch.utils.dlpack.from_dlpack(features.to_dlpack())
        self.weights = torch.utils.dlpack.from_dlpack(weights.to_dlpack())
        self.num_hits = torch.utils.dlpack.from_dlpack(num_hits.to_dlpack())

        return self.active_coordinates, self.features, self.weights, self.num_hits

    def insert(self, keys, new_feats, new_weights, new_num_hits):
        """[summary]

        Args:
            keys ([type]): [description]
            new_feats ([type]): [description]
            new_weights ([type]): [description]
            new_num_hits ([type]): [description]
        """
        if len(keys) == 0:
            return None

        o3c_keys = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(keys)).to(o3c.int64)
        feats_o3c = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(new_feats))
        weights_o3c = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(new_weights))
        num_hits_o3c = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(new_num_hits))
        buf_indices, masks_insert = self.indexer.insert(o3c_keys, (feats_o3c, weights_o3c, num_hits_o3c))
        if not masks_insert.cpu().numpy().all():
            existed_masks = masks_insert == False
            existed_keys = o3c_keys[existed_masks]
            buf_indices, masks_find = self.indexer.find(existed_keys)
            assert masks_find.cpu().numpy().all()
            self.indexer.value_tensor(0)[buf_indices.to(o3c.int64)] = feats_o3c[existed_masks]
            self.indexer.value_tensor(1)[buf_indices.to(o3c.int64)] = weights_o3c[existed_masks]
            self.indexer.value_tensor(2)[buf_indices.to(o3c.int64)] = num_hits_o3c[existed_masks]

    def reset(self, capacity):
        self.indexer = o3c.HashMap(
            capacity,
            key_dtype=o3c.int64,
            key_element_shape=(3,),
            value_dtypes=(o3c.Dtype.Float32, o3c.Dtype.Float32, o3c.Dtype.Float32),
            value_element_shapes=((self.n_feats,), (1,), (1,)),
            device=self.o3c_device)            
        # to be initialized in self.to_tensor
        self.tensor_indexer = None
        self.features = None
        self.weights = None
        self.num_hits = None
        self.active_coordinates = None

    def count_optim(self, keys):
        """[summary]

        Args:
            keys ([torch.Tensor]): shape: [1, 8, B, N, 3]

        Returns:
            [type]: [description]
        """
        shapes = [s for s in keys.shape]
        n_pts = np.asarray(shapes[:-1]).prod()
        assert shapes[-1] == 3
        o3c_keys = o3c.Tensor.from_dlpack(
            torch.utils.dlpack.to_dlpack(keys.reshape(-1, 3).long())
        )
        buf_indices, masks = self.tensor_indexer.find(o3c_keys)
        buf_indices = buf_indices[masks]
        indices = self.tensor_indexer.value_tensor()[buf_indices]
        indices = torch.utils.dlpack.from_dlpack(
            indices.to_dlpack())[:, 0]
        self.weights[indices] += 1


    def _query_tensor(self, keys):
        """[summary]

        Args:
            keys ([torch.Tensor]): shape: [1, 8, B, N, 3]

        Returns:
            [type]: [description]
        """
        shapes = [s for s in keys.shape]
        n_pts = np.asarray(shapes[:-1]).prod()
        assert shapes[-1] == 3
        out_feats = torch.zeros([n_pts, self.n_feats], device=self.device)
        out_weights = torch.zeros([n_pts, 1], device=self.device)
        out_num_hits = torch.zeros([n_pts, 1], device=self.device)

        o3c_keys = o3c.Tensor.from_dlpack(
            torch.utils.dlpack.to_dlpack(keys.reshape(-1, 3).long())
        )
        buf_indices, masks = self.tensor_indexer.find(o3c_keys)
        buf_indices = buf_indices[masks]
        indices = self.tensor_indexer.value_tensor()[buf_indices]
        indices = torch.utils.dlpack.from_dlpack(
            indices.to_dlpack())[:, 0]
        masks_torch = torch.utils.dlpack.from_dlpack(masks.to(o3c.int64).to_dlpack()).bool()
            
        out_feats[masks_torch] = self.features[indices]
        out_weights[masks_torch] = self.weights[indices]
        out_num_hits[masks_torch] = self.num_hits[indices]
        
        out_feats = out_feats.reshape(shapes[:-1] + [self.n_feats])
        out_weights = out_weights.reshape(shapes[:-1] + [1])
        out_num_hits = out_num_hits.reshape(shapes[:-1] + [1])

        return out_feats, out_weights, out_num_hits

    def query(self, keys):
        """[summary]

        Args:
            keys ([torch.Tensor]): shape: [..., 3]

        Returns:
            [type]: [description]
        """
        
        shapes = [s for s in keys.shape]
        n_pts = np.asarray(shapes[:-1]).prod()
        assert shapes[-1] == 3
        if n_pts == 0:
            return None, None, None
        
        out_feats = torch.zeros((n_pts, self.n_feats), device=self.device)
        out_weights = torch.zeros((n_pts, 1), device=self.device)
        out_num_hits = torch.zeros((n_pts, 1), device=self.device)
        
        o3c_keys = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(keys.reshape(-1, 3).long()))
        buf_inds, masks = self.indexer.find(o3c_keys)
        buf_inds = buf_inds[masks].to(o3c.int64)
        if not len(buf_inds) == 0:
            masks_torch = torch.utils.dlpack.from_dlpack(masks.to(o3c.int64).to_dlpack()).bool()
            out_feats[masks_torch] = torch.utils.dlpack.from_dlpack(
                self.indexer.value_tensor(0)[buf_inds].to_dlpack())
            out_weights[masks_torch] = torch.utils.dlpack.from_dlpack(
                self.indexer.value_tensor(1)[buf_inds].to_dlpack())
            out_num_hits[masks_torch] = torch.utils.dlpack.from_dlpack(
                self.indexer.value_tensor(2)[buf_inds].to_dlpack())
            out_feats = out_feats.reshape(shapes[:-1] + [self.n_feats])
            out_weights = out_weights.reshape(shapes[:-1] + [1])
            out_num_hits = out_num_hits.reshape(shapes[:-1] + [1])
        return out_feats, out_weights, out_num_hits

    def meshlize(self, nerf, sdf_delta=None, path=None):
        """ create mesh from the implicit volume

        Args:
            nerf ([type]): [description]
            path ([string]): the output mesh path
        """

        assert self.active_coordinates is not None, "call self.to_tensor() first."
        active_pts = self.active_coordinates * self.voxel_size + self.min_coords
        active_pts = active_pts.detach().cpu().numpy()
        active_coords = self.active_coordinates.detach().cpu().numpy()
        batch_size = 500
        step_size = 0.5
        level = 0.

        all_vertices = []
        all_faces = []
        last_face_id = 0

        for i in range(0, len(self.active_coordinates), batch_size):
            origin = active_coords[i: i + batch_size]
            n_batches = len(origin)
            range_ = np.arange(0, 1+step_size, step_size) - 0.5
            spacing = [range_[1] - range_[0]] * 3
            voxel_coords = np.stack(
                np.meshgrid(range_, range_, range_, indexing="ij"),
                axis=-1
            )
            voxel_coords = np.tile(voxel_coords, (n_batches, 1, 1, 1, 1))
            voxel_coords += origin[:, None, None, None, :]
            voxel_coords = torch.from_numpy(
                voxel_coords).float().to(self.device)
            H, W, D = voxel_coords.shape[1:4]
            voxel_coords = voxel_coords.reshape(1, n_batches, -1, 3)
            out = self.decode_pts(
                voxel_coords,
                nerf,
                sdf_delta,
                is_coords=True
            )
            sdf = out[0, :, :, 0].reshape(n_batches, H, W, D)
            sdf = sdf.detach().cpu().numpy()
            for j in range(n_batches):
                if np.max(sdf[j]) > level and np.min(sdf[j]) < level:
                    verts, faces, normals, values = \
                        marching_cubes(
                            sdf[j],
                            level=level,
                            spacing=spacing
                        )
                    verts += origin[j] - 0.5
                    all_vertices.append(verts)
                    all_faces.append(faces + last_face_id)
                    last_face_id += np.max(faces) + 1
        if len(all_vertices) == 0:
            return None
        final_vertices = np.concatenate(all_vertices, axis=0)
        final_faces = np.concatenate(all_faces, axis=0)
        final_vertices = final_vertices * self.voxel_size + self.min_coords.cpu().numpy()
        # all_normals = np.concatenate(all_normals, axis=0)
        mesh = trimesh.Trimesh(
            vertices=final_vertices,
            faces=final_faces,
            # vertex_normals=all_normals,
            process=False
        )
        if path is not None:
            mesh.export(path)
        return active_pts, mesh


    def decode_pts(
        self,
        coords,
        nerf,
        sdf_delta=None,
        is_coords=False,
        query_tensor=True,
    ):
        """ decode sdf values from the implicit volume given coords.

        Args:
            coords (_type_): [1, 8, n_pts, 3], input pts.
            nerf (_type_): _description_
            voxel_size (_type_): _description_
            sdf_delta (_type_, optional): _description_. Defaults to None.
            is_coords (_type_, optional): True if input pts are in voxel coords.
                Otherwise, they are in world coordinate that should be converted
                to voxel coords first.
            volume_resolution (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        if not is_coords:
            coords = (coords - self.min_coords) / self.voxel_size
        neighbor_coords = get_neighbors(coords)
        local_coords = coords.unsqueeze(1) - neighbor_coords
        assert torch.min(local_coords) >= -1
        assert torch.max(local_coords) <= 1
        weights_unmasked = torch.prod(
            1 - torch.abs(local_coords),
            dim=-1,
            keepdim=True
        )

        # get features from coords
        if query_tensor:
            feats, weights, num_hits = self._query_tensor(neighbor_coords)
        else:
            feats, weights, num_hits = self.query(neighbor_coords)
        mask = torch.min(weights, dim=1)[0] > self.min_pts_in_grid

        local_coords_encoded = nerf.xyz_encoding(local_coords)
        nerf_in = torch.cat([local_coords_encoded, feats], dim=-1)
        _, alpha = nerf.geo_forward(nerf_in)
        alpha = alpha * self.voxel_size
        normalizer = torch.sum(weights_unmasked, dim=1, keepdim=True)
        weights_unmasked = weights_unmasked / normalizer
        assert torch.all(torch.abs(weights_unmasked.sum(1) - 1) < 1e-5)
        alpha = torch.sum(alpha * weights_unmasked, dim=1)
        if sdf_delta is not None:
            neighbor_coords_grid_sample = neighbor_coords / (self.n_xyz-1)
            neighbor_coords_grid_sample = neighbor_coords_grid_sample * 2 - 1
            neighbor_coords_grid_sample = neighbor_coords_grid_sample[..., [2, 1, 0]]
            sdf_delta = F.grid_sample(
                sdf_delta,
                neighbor_coords_grid_sample,  # [1, 8, n_pts, n_steps, 3]
                mode="nearest",
                padding_mode="zeros",
                align_corners=True
            )
            sdf_delta = sdf_delta.permute(0, 2, 3, 4, 1)  # [B, 8, N, S, 1]
            sdf_delta = torch.sum(sdf_delta * weights_unmasked, dim=1)
            alpha += sdf_delta
        return alpha

    def save(self, path):
        self.print_statistic()
        active_buf_indices = self.tensor_indexer.active_buf_indices().to(o3c.int64)

        active_keys = self.tensor_indexer.key_tensor()[active_buf_indices]
        active_keys = torch.utils.dlpack.from_dlpack(active_keys.to_dlpack())
        
        active_vals = self.tensor_indexer.value_tensor()[active_buf_indices]
        active_vals = torch.utils.dlpack.from_dlpack(active_vals.to_dlpack())

        out_dict = {
            "25%": self.per_25 if self.per_25 else None,
            "50%": self.per_50,
            "75%": self.per_75,
            "dimensions": self.dimensions,
            "voxel_size": self.voxel_size,
            "mean": self.avg_n_pts,
            "min": self.min_pts,
            "active_keys": active_keys,
            "active_vals": active_vals,
            "features": self.features,
            "weights": self.weights,
            "num_hits": self.num_hits,
            "active_coordinates": self.active_coordinates
        }
        torch.save(out_dict, path + "_sparse_volume.pth")

    def load(self, path):
        volume = torch.load(path)
        active_keys = volume['active_keys']
        active_vals = volume['active_vals']
        features = volume['features']
        weights = volume['weights']
        num_hits = volume['num_hits']
        active_coordinates = volume['active_coordinates']

        self.tensor_indexer = o3c.HashMap(
            len(active_keys),
            key_dtype=o3c.int64,
            key_element_shape=(3,),
            value_dtype=o3c.int64,
            value_element_shape=(1,),
            device=o3c.Device(self.device)
        )
        active_keys = o3c.Tensor.from_dlpack(
            torch.utils.dlpack.to_dlpack(active_keys))
        active_vals = o3c.Tensor.from_dlpack(
            torch.utils.dlpack.to_dlpack(active_vals))

        buf_indices, masks = self.tensor_indexer.insert(
            active_keys, active_vals)
        masks = masks.cpu().numpy() if "cuda" in self.device else masks.numpy()
        assert masks.all()

        self.active_coordinates = active_coordinates
        self.features = features
        self.weights = weights
        self.num_hits = num_hits


class VolumeList:
    def __init__(self, n_feats, voxel_size, dimensions, min_pts_in_grid, capacity=100000, device="cuda:0") -> None:
        """
        Based on hash map implementation in Open3D.
        """
        # min_coords, max_coords, n_xyz = voxel_utils.get_world_range(
        #     dimensions, voxel_size)
        # self.coarse_min_coords = torch.from_numpy(min_coords).float().to(device)
        # self.coarse_max_coords = torch.from_numpy(max_coords).float().to(device)
        # self.coarse_n_xyz = torch.from_numpy(np.asarray(n_xyz)).long().to(device)
        # self.coarse_voxel_size = voxel_size
        # coarse_volume = SparseVolume(n_feats, n_xyz, min_coords, max_coords, min_pts_in_grid, capacity, device)
        # self.coarse_volume = coarse_volume
        # self.min_pts_in_grid = min_pts_in_grid

        fine_voxel_size = voxel_size
        min_coords, max_coords, n_xyz = voxel_utils.get_world_range(
            dimensions, fine_voxel_size)
        self.fine_min_coords = torch.from_numpy(min_coords).float().to(device)
        self.fine_max_coords = torch.from_numpy(max_coords).float().to(device)
        self.fine_n_xyz = torch.from_numpy(np.asarray(n_xyz)).long().to(device)
        self.fine_voxel_size = fine_voxel_size
        fine_volume = SparseVolume(n_feats, voxel_size, dimensions, min_pts_in_grid, capacity, device)
        self.fine_volume = fine_volume
        self.mesh_o3d_list = []

        self.device = device
        o3c_device = o3c.Device(device)

        # mesh of each voxel
        self.mesh_list = []
        self.mesh_indices = {}  # mapping from buf_indices to index of mesh_o3d_list

    def to_tensor(self):
        """ store all active values to pytorch tensor
        """
        # coarse_active_coords, coarse_feats, coarse_weights, coarse_num_hits = self.coarse_volume.to_tensor()
        fine_active_coords, fine_feats, fine_weights, fine_num_hits = self.fine_volume.to_tensor()
        # self.coarse_active_coords = coarse_active_coords
        # self.coarse_feats = coarse_feats
        # self.coarse_weights = coarse_weights
        # self.coarse_num_hits = coarse_num_hits
        
        self.fine_active_coords = fine_active_coords
        self.fine_feats = fine_feats
        self.fine_weights = fine_weights
        self.fine_num_hits = fine_num_hits

    def query(self, keys):
        """query value tensor stored in the hash map using keys. 

        Args:
            keys ([torch.Tensor]): size: [N, 3]. The coordinate ids of the query position.

        Returns:
            [type]: [description]
        """
        # feats_coarse, weights_coarse, num_hits_coarse = self.coarse_volume.query(keys[0])
        feats_fine, weights_fine, num_hits_fine = self.fine_volume.query(keys)
        return feats_fine, weights_fine, num_hits_fine

    def insert(self, keys, new_feats, new_weights, new_num_hits):
        """_summary_

        Args:
            keys (_type_): _description_
            new_feats (_type_): _description_
            new_weights (_type_): _description_
            new_num_hits (_type_): _description_
            visual (bool, optional): _description_. Defaults to False.
        """

        # self.coarse_volume.insert(keys[0], new_feats[0], new_weights[0], new_num_hits[0])
        self.fine_volume.insert(keys, new_feats, new_weights, new_num_hits)

    def meshlize_coords(self, coords, nerf, sdf_delta=None, volume_resolution=None):
        """_summary_

        Args:
            coords (_type_): [n_pts, 3]
            nerf (_type_): _description_
            sdf_delta (_type_, optional): _description_. Defaults to None.
            volume_resolution (_type_, optional): _description_. Defaults to None.
        """

        o3c_keys = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(coords.long()))
        buf_inds, masks = self.fine_volume.indexer.find(o3c_keys)
        buf_inds = buf_inds[masks].to(o3c.int64)
        buf_inds = torch.utils.dlpack.from_dlpack(buf_inds.to(o3c.int64).to_dlpack())
        masks_torch = torch.utils.dlpack.from_dlpack(masks.to(o3c.int64).to_dlpack()).bool()
        coords = coords[masks_torch]
        batch_size = 1000
        step_size = 0.5
        level = 0.
        for i in range(0, len(coords), batch_size):
            origin = coords[i: i + batch_size]
            n_batches = len(origin)
            range_ = np.arange(0, 1+step_size, step_size) - 0.5
            spacing = [range_[1] - range_[0]] * 3
            voxel_coords = np.stack(
                np.meshgrid(range_, range_, range_, indexing="ij"),
                axis=-1
            )
            voxel_coords = np.tile(voxel_coords, (n_batches, 1, 1, 1, 1))
            voxel_coords = torch.from_numpy(voxel_coords).float().to(self.device)
            voxel_coords += origin[:, None, None, None, :]
            pts = voxel_coords * self.fine_voxel_size + self.fine_min_coords
            H, W, D = pts.shape[1:4]
            pts = pts.reshape(1, n_batches, -1, 3)
            out = self.decode_pts(
                pts,
                nerf,
                sdf_delta,
                query_tensor=False
            )
            sdf = out[0, :, :, 0].reshape(n_batches, H, W, D)
            sdf = sdf.detach().cpu().numpy()
            for j in range(n_batches):
                if np.max(sdf[j]) > level and np.min(sdf[j]) < level:
                    verts, faces, normals, values = \
                        marching_cubes(
                            sdf[j],
                            level=level,
                            spacing=spacing
                        )
                    verts += origin[j].cpu().numpy() - 0.5
                    mesh = trimesh.Trimesh(
                        vertices=verts,
                        faces=faces,
                        vertex_normals=normals,
                        process=False
                    )
                    # update or create a new mesh for each voxel
                    if buf_inds[j] not in self.mesh_indices:
                        self.mesh_indices[buf_inds[j]] = len(self.mesh_list)
                        self.mesh_list.append(mesh)
                    else:
                        self.mesh_list[self.mesh_indices[buf_inds[j]]] = mesh

    def merge_meshes(self):
        all_vertices = []
        all_faces = []
        last_face_id = 0
        for i in range(len(self.mesh_list)):
            mesh = self.mesh_list[i]
            all_vertices.append(mesh.vertices)
            all_faces.append(mesh.faces + last_face_id)
            last_face_id += np.max(mesh.faces) + 1
        final_vertices = np.concatenate(all_vertices, axis=0)
        final_faces = np.concatenate(all_faces, axis=0)
        final_vertices = final_vertices * self.fine_voxel_size + self.fine_min_coords.cpu().numpy()
        return trimesh.Trimesh(
            vertices=final_vertices,
            faces=final_faces
        )

    def meshlize(self, nerf, sdf_delta=None, volume_resolution=None, path=None):
        """ create mesh from the implicit volume

        Args:
            nerf ([type]): [description]
            path ([string]): the output mesh path
        """
        # coarse_active_coords = self.coarse_volume.active_coordinates
        # mask = self.coarse_volume.weights > self.coarse_volume.min_pts_in_grid

        active_coords = self.fine_volume.active_coordinates
        active_pts = active_coords * self.fine_voxel_size + self.fine_min_coords
        active_pts = active_pts.detach().cpu().numpy()
        active_coords = active_coords.detach().cpu().numpy()
        batch_size = 500
        step_size = 0.5
        level = 0.

        all_vertices = []
        all_faces = []
        last_face_id = 0

        for i in range(0, len(active_coords), batch_size):
            origin = active_coords[i: i + batch_size]
            n_batches = len(origin)
            range_ = np.arange(0, 1+step_size, step_size) - 0.5
            spacing = [range_[1] - range_[0]] * 3
            voxel_coords = np.stack(
                np.meshgrid(range_, range_, range_, indexing="ij"),
                axis=-1
            )
            voxel_coords = np.tile(voxel_coords, (n_batches, 1, 1, 1, 1))
            voxel_coords += origin[:, None, None, None, :]
            voxel_coords = torch.from_numpy(voxel_coords).float().to(self.device)
            pts = voxel_coords * self.fine_voxel_size + self.fine_min_coords
            H, W, D = pts.shape[1:4]
            pts = pts.reshape(1, n_batches, -1, 3)
            out = self.decode_pts(
                pts,
                nerf,
                sdf_delta,
            )
            sdf = out[0, :, :, 0].reshape(n_batches, H, W, D)
            sdf = sdf.detach().cpu().numpy()
            for j in range(n_batches):
                if np.max(sdf[j]) > level and np.min(sdf[j]) < level:
                    verts, faces, normals, values = \
                        marching_cubes(
                            sdf[j],
                            level=level,
                            spacing=spacing
                        )
                    verts += origin[j] - 0.5
                    all_vertices.append(verts)
                    all_faces.append(faces + last_face_id)
                    last_face_id += np.max(faces) + 1
        if len(all_vertices) == 0:
            return None
        final_vertices = np.concatenate(all_vertices, axis=0)
        final_faces = np.concatenate(all_faces, axis=0)
        final_vertices = final_vertices * self.fine_voxel_size + self.fine_min_coords.cpu().numpy()
        # all_normals = np.concatenate(all_normals, axis=0)
        mesh = trimesh.Trimesh(
            vertices=final_vertices,
            faces=final_faces,
            # vertex_normals=all_normals,
            process=False
        )
        if path is not None:
            mesh.export(path)
        return active_pts, mesh

    def decode_pts(
        self,
        pts,
        nerf,
        sdf_delta=None,
        query_tensor=True
    ):
        """decode sdf values from the implicit volume given coords.

        Args:
            pts (_type_): _description_
            nerf (_type_): _description_
            sdf_delta (List, optional): _description_. Defaults to None.
            query_tensor (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        fine_coords = (pts - self.fine_min_coords) / self.fine_voxel_size
        alpha_fine = self.fine_volume.decode_pts(
            fine_coords,
            nerf,
            sdf_delta=sdf_delta[0] if sdf_delta is not None else None,
            is_coords=True,
            query_tensor=query_tensor
        )
        alpha = alpha_fine
        return alpha

    def save(self, path):
        # self.coarse_volume.save(path + "_coarse")
        self.fine_volume.save(path + "_fine")

    def load(self, path):
        # self.coarse_volume.load(path[0])
        self.fine_volume.load(path)


if __name__ == "__main__":
    o3c_device = o3c.Device("cuda:0")
    th_a = torch.ones((5, 1), device="cuda", requires_grad=True)

    optimizer = torch.optim.SGD(
        [{'params': [th_a]}],
        lr=1e-2,
        momentum=0.9)

    volume = SparseVolume(10, 1)

    keys = o3c.Tensor([[0], [1], [2], [3], [4]],
                    dtype=o3c.int64,
                    device=o3c_device)
    vals = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(th_a))
    volume.hashmap.insert(keys, vals)


    for i in range(100):
        active_keys = i % 5
        query_keys = o3c.Tensor(
            [[active_keys]],
            dtype=o3c.int64,
            device=o3c_device)
        buf_indices, masks = volume.hashmap.find(query_keys)
        buf_indices = buf_indices[masks].to(o3c.int64)
        valid_vals = volume.hashmap.value_tensor()[buf_indices]

        o3_a = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(th_a))

    device = o3c.Device("cuda:0")
    buf_indices, masks = volume.insert(keys, vals)

