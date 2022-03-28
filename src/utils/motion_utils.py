import numpy as np
from pyquaternion import Quaternion


def so3_vee(Phi):
    if Phi.ndim < 3:
        Phi = np.expand_dims(Phi, axis=0)

    if Phi.shape[1:3] != (3, 3):
        raise ValueError("Phi must have shape ({},{}) or (N,{},{})".format(3, 3, 3, 3))

    phi = np.empty([Phi.shape[0], 3])
    phi[:, 0] = Phi[:, 2, 1]
    phi[:, 1] = Phi[:, 0, 2]
    phi[:, 2] = Phi[:, 1, 0]
    return np.squeeze(phi)


def so3_wedge(phi):
    phi = np.atleast_2d(phi)
    if phi.shape[1] != 3:
        raise ValueError(
            "phi must have shape ({},) or (N,{})".format(3, 3))

    Phi = np.zeros([phi.shape[0], 3, 3])
    Phi[:, 0, 1] = -phi[:, 2]
    Phi[:, 1, 0] = phi[:, 2]
    Phi[:, 0, 2] = phi[:, 1]
    Phi[:, 2, 0] = -phi[:, 1]
    Phi[:, 1, 2] = -phi[:, 0]
    Phi[:, 2, 1] = phi[:, 0]
    return np.squeeze(Phi)


def so3_log(matrix):
    cos_angle = 0.5 * np.trace(matrix) - 0.5
    cos_angle = np.clip(cos_angle, -1., 1.)
    angle = np.arccos(cos_angle)
    if np.isclose(angle, 0.):
        return so3_vee(matrix - np.identity(3))
    else:
        return so3_vee((0.5 * angle / np.sin(angle)) * (matrix - matrix.T))


def so3_left_jacobian(phi):
    angle = np.linalg.norm(phi)

    if np.isclose(angle, 0.):
        return np.identity(3) + 0.5 * so3_wedge(phi)

    axis = phi / angle
    s = np.sin(angle)
    c = np.cos(angle)

    return (s / angle) * np.identity(3) + \
           (1 - s / angle) * np.outer(axis, axis) + \
           ((1 - c) / angle) * so3_wedge(axis)


def se3_curlywedge(xi):
    xi = np.atleast_2d(xi)

    Psi = np.zeros([xi.shape[0], 6, 6])
    Psi[:, 0:3, 0:3] = so3_wedge(xi[:, 3:6])
    Psi[:, 0:3, 3:6] = so3_wedge(xi[:, 0:3])
    Psi[:, 3:6, 3:6] = Psi[:, 0:3, 0:3]

    return np.squeeze(Psi)


def se3_left_jacobian_Q_matrix(xi):
    rho = xi[0:3]  # translation part
    phi = xi[3:6]  # rotation part

    rx = so3_wedge(rho)
    px = so3_wedge(phi)

    ph = np.linalg.norm(phi)
    ph2 = ph * ph
    ph3 = ph2 * ph
    ph4 = ph3 * ph
    ph5 = ph4 * ph

    cph = np.cos(ph)
    sph = np.sin(ph)

    m1 = 0.5
    m2 = (ph - sph) / ph3
    m3 = (0.5 * ph2 + cph - 1.) / ph4
    m4 = (ph - 1.5 * sph + 0.5 * ph * cph) / ph5

    t1 = rx
    t2 = px.dot(rx) + rx.dot(px) + px.dot(rx).dot(px)
    t3 = px.dot(px).dot(rx) + rx.dot(px).dot(px) - 3. * px.dot(rx).dot(px)
    t4 = px.dot(rx).dot(px).dot(px) + px.dot(px).dot(rx).dot(px)

    return m1 * t1 + m2 * t2 + m3 * t3 + m4 * t4


def se3_left_jacobian(xi):
    rho = xi[0:3]  # translation part
    phi = xi[3:6]  # rotation part

    # Near |phi|==0, use first order Taylor expansion
    if np.isclose(np.linalg.norm(phi), 0.):
        return np.identity(6) + 0.5 * se3_curlywedge(xi)

    so3_jac = so3_left_jacobian(phi)
    Q_mat = se3_left_jacobian_Q_matrix(xi)

    jac = np.zeros([6, 6])
    jac[0:3, 0:3] = so3_jac
    jac[0:3, 3:6] = Q_mat
    jac[3:6, 3:6] = so3_jac

    return jac


def se3_inv_left_jacobian(xi):
    rho = xi[0:3]  # translation part
    phi = xi[3:6]  # rotation part

    # Near |phi|==0, use first order Taylor expansion
    if np.isclose(np.linalg.norm(phi), 0.):
        return np.identity(6) - 0.5 * se3_curlywedge(xi)

    so3_inv_jac = so3_inv_left_jacobian(phi)
    Q_mat = se3_left_jacobian_Q_matrix(xi)

    jac = np.zeros([6, 6])
    jac[0:3, 0:3] = so3_inv_jac
    jac[0:3, 3:6] = -so3_inv_jac.dot(Q_mat).dot(so3_inv_jac)
    jac[3:6, 3:6] = so3_inv_jac

    return jac


def so3_inv_left_jacobian(phi):
    angle = np.linalg.norm(phi)

    if np.isclose(angle, 0.):
        return np.identity(3) - 0.5 * so3_wedge(phi)

    axis = phi / angle
    half_angle = 0.5 * angle
    cot_half_angle = 1. / np.tan(half_angle)

    return half_angle * cot_half_angle * np.identity(3) + \
           (1 - half_angle * cot_half_angle) * np.outer(axis, axis) - \
           half_angle * so3_wedge(axis)


def project_orthogonal(rot):
    u, s, vh = np.linalg.svd(rot, full_matrices=True, compute_uv=True)
    rot = u @ vh
    if np.linalg.det(rot) < 0:
        u[:, 2] = -u[:, 2]
        rot = u @ vh
    return rot


class Isometry:
    GL_POST_MULT = Quaternion(degrees=180.0, axis=[1.0, 0.0, 0.0])

    def __init__(self, q=None, t=None):
        if q is None:
            q = Quaternion()
        if t is None:
            t = np.zeros(3)
        if not isinstance(t, np.ndarray):
            t = np.asarray(t)
        assert t.shape[0] == 3 and t.ndim == 1
        self.q = q
        self.t = t

    def __repr__(self):
        return f"Isometry: t = {self.t}, q = {self.q}"

    @property
    def rotation(self):
        return Isometry(q=self.q)

    @property
    def matrix(self):
        mat = self.q.transformation_matrix
        mat[0:3, 3] = self.t
        return mat

    @staticmethod
    def from_matrix(mat, t_component=None, ortho=False):
        assert isinstance(mat, np.ndarray)
        if t_component is None:
            assert mat.shape == (4, 4)
            if ortho:
                mat[:3, :3] = project_orthogonal(mat[:3, :3])
            return Isometry(q=Quaternion(matrix=mat), t=mat[:3, 3])
        else:
            assert mat.shape == (3, 3)
            assert t_component.shape == (3,)
            if ortho:
                mat = project_orthogonal(mat)
            return Isometry(q=Quaternion(matrix=mat), t=t_component)

    @staticmethod
    def from_twist(xi: np.ndarray):
        rho = xi[:3]
        phi = xi[3:6]
        iso = Isometry.from_so3_exp(phi)
        iso.t = so3_left_jacobian(phi) @ rho
        return iso

    @staticmethod
    def from_so3_exp(phi: np.ndarray):
        angle = np.linalg.norm(phi)

        # Near phi==0, use first order Taylor expansion
        if np.isclose(angle, 0.):
            return Isometry(q=Quaternion(matrix=np.identity(3) + so3_wedge(phi)))

        axis = phi / angle
        s = np.sin(angle)
        c = np.cos(angle)

        rot_mat = (c * np.identity(3) +
                   (1 - c) * np.outer(axis, axis) +
                   s * so3_wedge(axis))
        return Isometry(q=Quaternion(matrix=rot_mat))

    @property
    def continuous_repr(self):
        rot = self.q.rotation_matrix[:, 0:2].T.flatten()    # (6,)
        return np.concatenate([rot, self.t])                # (9,)

    @staticmethod
    def from_continuous_repr(rep, gs=True):
        if isinstance(rep, list):
            rep = np.asarray(rep)
        assert isinstance(rep, np.ndarray)
        assert rep.shape == (9,)
        # For rotation, use Gram-Schmidt orthogonalization
        col1 = rep[0:3]
        col2 = rep[3:6]
        if gs:
            col1 /= np.linalg.norm(col1)
            col2 = col2 - np.dot(col1, col2) * col1
            col2 /= np.linalg.norm(col2)
        col3 = np.cross(col1, col2)
        return Isometry(q=Quaternion(matrix=np.column_stack([col1, col2, col3])), t=rep[6:9])

    @property
    def full_repr(self):
        rot = self.q.rotation_matrix.T.flatten()
        return np.concatenate([rot, self.t])

    @staticmethod
    def from_full_repr(rep, ortho=False):
        assert isinstance(rep, np.ndarray)
        assert rep.shape == (12,)
        rot = rep[0:9].reshape(3, 3).T
        if ortho:
            rot = project_orthogonal(rot)
        return Isometry(q=Quaternion(matrix=rot), t=rep[9:12])

    def torch_matrices(self, device):
        import torch
        return torch.from_numpy(self.q.rotation_matrix).to(device).float(), \
               torch.from_numpy(self.t).to(device).float()

    @staticmethod
    def random():
        return Isometry(q=Quaternion.random(), t=np.random.random((3,)))

    def inv(self):
        qinv = self.q.inverse
        return Isometry(q=qinv, t=-(qinv.rotate(self.t)))

    def dot(self, right):
        return Isometry(q=(self.q * right.q), t=(self.q.rotate(right.t) + self.t))

    def to_gl_camera(self):
        return Isometry(q=(self.q * self.GL_POST_MULT), t=self.t)

    @staticmethod
    def look_at(source: np.ndarray, target: np.ndarray, up: np.ndarray = None):
        z_dir = target - source
        z_dir /= np.linalg.norm(z_dir)
        if up is None:
            up = np.asarray([0.0, 1.0, 0.0])
            if np.linalg.norm(np.cross(z_dir, up)) < 1e-6:
                up = np.asarray([1.0, 0.0, 0.0])
        else:
            up /= np.linalg.norm(up)
        x_dir = np.cross(z_dir, up)
        x_dir /= np.linalg.norm(x_dir)
        y_dir = np.cross(z_dir, x_dir)
        R = np.column_stack([x_dir, y_dir, z_dir])
        return Isometry(q=Quaternion(matrix=R), t=source)

    def adjoint_matrix(self):
        R = self.q.rotation_matrix
        twR = so3_wedge(self.t) @ R
        adjoint = np.zeros((6, 6))
        adjoint[0:3, 0:3] = R
        adjoint[3:6, 3:6] = R
        adjoint[0:3, 3:6] = twR
        return adjoint

    def log(self):
        phi = so3_log(self.q.rotation_matrix)
        rho = so3_inv_left_jacobian(phi) @ self.t
        return np.hstack([rho, phi])

    def tangent(self, prev_iso, next_iso):
        t = 0.5 * (next_iso.t - prev_iso.t)
        l1 = Quaternion.log((self.q.inverse * prev_iso.q).normalised)
        l2 = Quaternion.log((self.q.inverse * next_iso.q).normalised)
        e = Quaternion()
        e.q = -0.25 * (l1.q + l2.q)
        e = self.q * Quaternion.exp(e)
        return Isometry(t=t, q=e)

    def __matmul__(self, other):
        # "@" operator: other can be (N,3) or (3,).
        if hasattr(other, "device"):        # Torch tensor
            assert other.ndim == 2 and other.size(1) == 3  # (N,3)
            th_R, th_t = self.torch_matrices(other.device)
            return other @ th_R.t() + th_t.unsqueeze(0)
        if isinstance(other, Isometry):
            return self.dot(other)
        if type(other) != np.ndarray or other.ndim == 1:
            return self.q.rotate(other) + self.t
        else:
            return other @ self.q.rotation_matrix.T + self.t[np.newaxis, :]

    @staticmethod
    def interpolate(source, target, alpha):
        iquat = Quaternion.slerp(source.q, target.q, alpha)
        it = source.t * (1 - alpha) + target.t * alpha
        return Isometry(q=iquat, t=it)
