import torch


def random_face(mesh: torch.Tensor, num_samples: int, distrib=None):
    """Return an area weighted random sample of faces and their normals from the mesh.

    Args:
        mesh (torch.Tensor): #F, 3, 3 array of vertices
        num_samples (int): num of samples to return
        distrib: distribution to use. By default, area-weighted distribution is used.
    """

    if distrib is None:
        distrib = area_weighted_distribution(mesh)

    normals = per_face_normals(mesh)

    idx = distrib.sample([num_samples])

    return mesh[idx], normals[idx]


def per_face_normals(mesh: torch.Tensor):
    """Compute normals per face.

    Args:
        mesh (torch.Tensor): #F, 3, 3 array of vertices
    """

    vec_a = mesh[:, 0] - mesh[:, 1]
    vec_b = mesh[:, 1] - mesh[:, 2]
    normals = torch.cross(vec_a, vec_b)
    return normals


def area_weighted_distribution(
    mesh: torch.Tensor, normals: torch.Tensor = None
):
    """Construct discrete area weighted distribution over triangle mesh.

    Args:
        mesh (torch.Tensor): #F, 3, 3 array of vertices
        normals (torch.Tensor): normals (if precomputed)
        eps (float): epsilon
    """

    if normals is None:
        normals = per_face_normals(mesh)
    areas = torch.norm(normals, p=2, dim=1) * 0.5
    areas /= torch.sum(areas) + 1e-10

    # Discrete PDF over triangles
    return torch.distributions.Categorical(areas.view(-1))


def sample_near_surface(mesh: torch.Tensor, num_samples: int, distrib=None):
    """Sample points near the mesh surface.

    Args:
        mesh (torch.Tensor): triangle mesh
        num_samples (int): number of surface samples
        distrib: distribution to use. By default, area-weighted distribution is used
    """
    if distrib is None:
        distrib = area_weighted_distribution(mesh)
    samples = sample_surface(mesh, num_samples, distrib)[0]
    samples += torch.randn_like(samples) * 0.01
    return samples


def sample_uniform(num_samples: int):
    """Sample uniformly in [-1,1] bounding volume.

    Args:
        num_samples(int) : number of points to sample
    """
    return torch.rand(num_samples, 3) * 2.0 - 1.0


def sample_surface(
    mesh: torch.Tensor,
    num_samples: int,
    distrib=None,
):
    """Sample points and their normals on mesh surface.

    Args:
        mesh (torch.Tensor): triangle mesh
        num_samples (int): number of surface samples
        distrib: distribution to use. By default, area-weighted distribution is used
    """
    if distrib is None:
        distrib = area_weighted_distribution(mesh)

    # Select faces & sample their surface
    f, normals = random_face(mesh, num_samples, distrib)

    u = torch.sqrt(torch.rand(num_samples)).to(mesh.device).unsqueeze(-1)
    v = torch.rand(num_samples).to(mesh.device).unsqueeze(-1)

    samples = (1 - u) * f[:, 0, :] + (u * (1 - v)) * f[:, 1, :] + u * v * f[:, 2, :]

    return samples, normals


def point_sample(mesh: torch.Tensor, techniques: list, num_samples: int):
    """Sample points from a mesh.

    Args:
        mesh (torch.Tensor): #F, 3, 3 array of vertices
        techniques (list[str]): list of techniques to sample with
        num_samples (int): points to sample per technique
    """
    if 'trace' in techniques or 'near' in techniques:
        # Precompute face distribution
        distrib = area_weighted_distribution(mesh)

    samples = []
    for technique in techniques:
        if technique == 'trace':
            samples.append(sample_surface(mesh, num_samples, distrib)[0])
        elif technique == 'near':
            samples.append(sample_near_surface(mesh, num_samples, distrib))
        elif technique == 'rand':
            samples.append(sample_uniform(num_samples).to(mesh.device))
    samples = torch.cat(samples, dim=0)
    return samples