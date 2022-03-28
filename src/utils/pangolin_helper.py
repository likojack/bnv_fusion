import pangolin
import OpenGL.GL as gl
import numpy as np


def init_panel():
    pangolin.ParseVarsFile('app.cfg')

    pangolin.CreateWindowAndBind('Main', 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000),
        pangolin.ModelViewLookAt(0, 0.5, -3, 0, 0, 0, pangolin.AxisDirection.AxisY))
    handler3d = pangolin.Handler3D(scam)

    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 180/640., 1.0, -640.0/480.0)
    # dcam.SetBounds(pangolin.Attach(0.0),     pangolin.Attach(1.0), 
                     # pangolin.Attach.Pix(180), pangolin.Attach(1.0), -640.0/480.0)

    dcam.SetHandler(pangolin.Handler3D(scam))

    panel = pangolin.CreatePanel('ui')
    panel.SetBounds(0.0, 1.0, 0.0, 180/640.)
    return scam, dcam


def draw_3d_points(display, scene_camera, points, colors, pt_size=5):
    """display 3d point cloud

    Args:
        display (Pangolin object): display window
        scene_camera (Pangolin object): scene camera
        points (np.ndarray): [N, 3] 3D point position
        colors (np.ndarray): [N, 3] color for each point, range [0, 1]
        pt_size (int): point size
    Returns:
        None
    """
    display.Activate(scene_camera)
    gl.glPointSize(pt_size)
    gl.glColor3f(1.0, 0.0, 0.0)
    pangolin.DrawPoints(points, colors)


def draw_3d_box(display, scene_camera, rgb_color, t_wo, dimensions, line_width=3, kitti=False, alpha=1):
    """display object 3d bounding box

    Args:
        display (Pangolin object): display window
        scene_camera (Pangolin object): scene camera
        rgb_color (np.ndarray): [3, ] box color, range [0, 1]
        t_wo (np.ndarray): [4, 4] transform matrix from object to world coordinate
        dimensions (np.ndarray): [3] object dimension
        line_width: line with of bounding box
        kitti: if use kitti format
    kitti use a weird format for bbox_3d and translation
    the translation is not the object bounding box center, but the (x_mean, y_min, z_mean)
    and although the dimension is for h, w, l, it doesn't align with the coordinate order.
    h is for y, w is for z, and l is for x. So we need to do two things to align with
    representation. (1) switch the order of dimensions, (2) minus t_y so that it is in the
    translation is in the object center.
    Returns:
        None
    """
    if kitti:
        t_wo = deepcopy(t_wo)
        t_wo[1, 3] -= dimensions[1] / 2.
    display.Activate(scene_camera)
    gl.glLineWidth(line_width)
    gl.glColor4f(rgb_color[0], rgb_color[1], rgb_color[2], alpha)
    pangolin.DrawBoxes([t_wo], dimensions[None, :])


def draw_mesh(display, scene_camera, points, faces, normals, alpha_value=0.6):
    """display mesh

    Args:
        display (Pangolin object): display window
        scene_camera (Pangolin object): scene camera
        points (np.ndarray): [N, 3] 3D point position
        faces (np.ndarray): [NUM_F, 3] mesh faces
        normals (np.ndarray): [N, 4] vertex normal
        colors (np.ndarray): [N, 3] color for each point, range [0, 1]
    Returns:
        None
    """
    alpha = np.ones((len(normals), 1)) * alpha_value
    smoothed_normals = np.concatenate([normals, alpha], axis=1)
    display.Activate(scene_camera)
    pangolin.DrawMesh(points, faces.astype(np.int32), smoothed_normals)


def draw_line(display, scene_camera, points, colors, line_width=5, alpha=1):
    """display line in 3D display

    Args:
        display (Pangolin object): display window
        scene_camera (Pangolin object): scene camera
        points (np.ndarray): [2, 3] points positions on the line
        colors (np.ndarray): [1, 3] lien color, range [0, 1]
    Returns:
        None
    """

    display.Activate(scene_camera)
    gl.glLineWidth(line_width)
    gl.glColor4f(colors[0], colors[1], colors[2], colors[3])
    pangolin.DrawLine(points)


def draw_lines(display, scene_camera, lines, colors, line_width=5):
    """display line in 3D display

    Args:
        display (Pangolin object): display window
        scene_camera (Pangolin object): scene camera
        lines (np.ndarray): [n_lines, 2, 3] points positions on the line
        colors (np.ndarray): [3] lien color, range [0, 1]
    Returns:
        None
    """

    display.Activate(scene_camera)
    for line in lines:
        gl.glLineWidth(line_width)
        gl.glColor3f(colors[0], colors[1], colors[2])
        pangolin.DrawLine(line)


def draw_image(rgb, texture, display):
    """display image

    Args:
        rgb (np.ndarray): [H, W, 3] rgb image
        texture (Pangolin object): see pangolin
        display (Pangolin object): display camera
    Returns:
        None
    """
    texture.Upload(rgb, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    display.Activate()
    gl.glColor3f(1.0, 1.0, 1.0)
    texture.RenderToViewport()