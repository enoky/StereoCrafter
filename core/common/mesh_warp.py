#!/usr/bin/env python3
"""
Fusion3DStereoReplicator - True 3D stereoscopic rendering in Python.

This module perfectly replicates DaVinci Resolve Fusion's 3D stereo workflow.
Instead of 2D pixel-shifting, it uses a headless ModernGL engine to:
1. Generate a dense 3D continuous geometric plane.
2. Extrude vertices in 3D space toward an extrusion point (shearing/dolly).
3. Set up two parallel Off-Axis virtual cameras.
4. Render the Left and Right eye views to eliminate 'hole-filling' artifacts
   by allowing natural 3D triangle stretching.

Designed to process single frames/depthmaps, but fast enough (GPU hardware)
to be easily wrapped in a video processing loop.
"""

import numpy as np
import cv2
import moderngl
from typing import Tuple


class FusionStereoReplicator:
    """
    A headless hardware-accelerated 3D stereo rendering engine.

    Parameters
    ----------
    width : int
        The width of the input/output images.
    height : int
        The height of the input/output images.
    density_x : int, optional
        Number of vertices across the X-axis of the ImagePlane3D (default: 512).
    density_y : int, optional
        Number of vertices across the Y-axis of the ImagePlane3D (default: 512).
    """

    def __init__(self, width, height, density_x=512, density_y=512):
        self.width = width
        self.height = height
        # 1. Initialize Headless OpenGL Context
        self.ctx = moderngl.create_context(standalone=True)
        self.ctx.enable(moderngl.DEPTH_TEST)

        # Ensure memory alignment is byte-perfect to avoid stride issues at unusual resolutions
        self.ctx.pack_alignment = 1
        self.ctx.unpack_alignment = 1

        # 2. GLSL Shaders
        vertex_shader = """
            #version 330
            in vec2 in_uv;
            out vec2 v_uv;

            uniform sampler2D tex_depth;

            uniform float img_width;
            uniform float img_height;
            uniform vec3 extrusion_point;
            uniform float extrusion_scale;

            uniform mat4 View;
            uniform mat4 Proj;

            void main() {
                v_uv = in_uv;

                // Read depth (0.0 to 1.0)
                float D = texture(tex_depth, in_uv).r;

                // Construct base vertex in world coordinates at Z = 0
                vec3 V_base = vec3((in_uv.x - 0.5) * img_width, (in_uv.y - 0.5) * img_height, 0.0);

                // Extrude / Shear towards the defined point in space
                vec3 dir = extrusion_point - V_base;
                vec3 V_new = V_base + dir * (D * extrusion_scale);

                gl_Position = Proj * View * vec4(V_new, 1.0);
            }
        """

        fragment_shader = """
            #version 330
            in vec2 v_uv;
            out vec3 f_color;

            uniform sampler2D tex_color;

            void main() {
                f_color = texture(tex_color, v_uv).rgb;
            }
        """
        self.prog = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

        # 3. Generate the 3D grid mesh (Vertices and Indices)
        x = np.linspace(0, 1, density_x, dtype=np.float32)
        y = np.linspace(0, 1, density_y, dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        uvs = np.column_stack((X.ravel(), Y.ravel()))

        indices = []
        for j in range(density_y - 1):
            for i in range(density_x - 1):
                p0 = j * density_x + i
                p1 = p0 + 1
                p2 = (j + 1) * density_x + i
                p3 = p2 + 1
                indices.extend([p0, p2, p1])
                indices.extend([p1, p2, p3])
        indices = np.array(indices, dtype=np.int32)

        # Upload mesh to GPU
        self.vbo = self.ctx.buffer(uvs.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "2f", "in_uv")], self.ibo)

        # 4. Setup Textures and Framebuffers
        self.tex_color = self.ctx.texture((width, height), 3, dtype="f1")
        self.tex_depth = self.ctx.texture((width, height), 1, dtype="f4")

        self.tex_out_left = self.ctx.texture((width, height), 3, dtype="f1")
        self.fbo_left = self.ctx.framebuffer([self.tex_out_left], self.ctx.depth_texture((width, height)))

        self.tex_out_right = self.ctx.texture((width, height), 3, dtype="f1")
        self.fbo_right = self.ctx.framebuffer([self.tex_out_right], self.ctx.depth_texture((width, height)))

    @staticmethod
    def _perspective_off_axis(left, right, bottom, top, near, far):
        """Constructs a mathematically exact Off-Axis Projection Matrix."""
        M = np.zeros((4, 4), dtype=np.float32)
        M[0, 0] = (2.0 * near) / (right - left)
        M[1, 1] = (2.0 * near) / (top - bottom)
        M[0, 2] = (right + left) / (right - left)
        M[1, 2] = (top + bottom) / (top - bottom)
        M[2, 2] = -(far + near) / (far - near)
        M[2, 3] = -(2.0 * far * near) / (far - near)
        M[3, 2] = -1.0
        return M

    @staticmethod
    def _view_matrix(cx, cy, cz):
        """Constructs a standard View Matrix for a camera looking down -Z."""
        M = np.eye(4, dtype=np.float32)
        M[0, 3] = -cx
        M[1, 3] = -cy
        M[2, 3] = -cz
        return M

    def process_frame(
        self,
        image,
        depth_map,
        disparity=25.0,
        convergence=0.5,
        view_bias=0.0,
        dolly_zoom=0.0,
        extrusion_scale=0.1,
        camera_distance=None,
    ):
        """
        Renders a single frame into Left and Right stereo pairs.

        Parameters
        ----------
        image : np.ndarray
            RGB or BGR image (H, W, 3).
        depth_map : np.ndarray
            Grayscale depth map (H, W).
        disparity : float, optional
            Maximum pixel shift represented as (Percentage of screen width * 10).
            E.g., 25 = 2.5% of the screen width. Default is 25.0.
        convergence : float, optional
            Where the views converge (zero parallax).
            0.0 = Extruded Max Foreground (White pixels)
            1.0 = Base Background (Black pixels).
        view_bias : float, optional
            Shifts the extrusion apex horizontally.
            -1.0 = Apex at Left Camera, 0.0 = Centered, 1.0 = Apex at Right Camera.
        dolly_zoom : float, optional
            Z-offset for the extrusion apex relative to the camera distance.
        extrusion_scale : float, optional
            How much of the distance between the plane and the extrusion_point
            is covered by the maximum depth value. Default is 0.1 (10% of the way).
        camera_distance : float, optional
            Distance of the virtual cameras from the image plane.
            Defaults to width * 1.5.

        Returns
        -------
        left_img, right_img : np.ndarray
            The rendered Left and Right BGR images.
        """
        # Ensure incoming data matches the initialized texture size
        if image.shape[1] != self.width or image.shape[0] != self.height:
            image = cv2.resize(image, (self.width, self.height))

        if depth_map.shape[1] != self.width or depth_map.shape[0] != self.height:
            depth_map = cv2.resize(depth_map, (self.width, self.height))

        # Handle various image formats (RGB, BGR, RGBA)
        if image.ndim == 3:
            channels = image.shape[2]
            if channels == 4:  # RGBA -> RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif channels == 3:  # BGR -> RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        depth_float = depth_map.astype(np.float32)
        if depth_float.max() > 1.0:
            depth_float /= 255.0  # Normalize to 0-1

        # OpenGL expects bottom-left origin
        image_rgb = cv2.flip(image, 0)
        depth_float = cv2.flip(depth_float, 0)

        # CRITICAL: Ensure memory is contiguous before calling .tobytes()
        # Non-contiguous arrays (like results of cv2.flip or transpose) will return
        # corrupted memory buffers via .tobytes().
        image_bytes = np.ascontiguousarray(image_rgb, dtype=np.uint8).tobytes()
        depth_bytes = np.ascontiguousarray(depth_float, dtype=np.float32).tobytes()

        self.tex_color.write(image_bytes)
        self.tex_depth.write(depth_bytes)

        # 1. Setup World Parameters
        if camera_distance is None:
            camera_distance = self.width * 1.5

        # Calculate Disparity in 3D world units (Percentage / 10 * 100 = Percentage / 1000)
        disparity_world = self.width * (disparity / 1000.0)

        # Calculate Absolute Extrusion Point
        # X is biased between eyes: -1 = Left ( -disparity_world / 2 ), 1 = Right ( +disparity_world / 2 )
        abs_x = view_bias * (disparity_world / 2.0)
        abs_y = 0.0  # Vertical bias not supported in standard stereo
        abs_z = camera_distance + dolly_zoom
        abs_extrusion_point = (abs_x, abs_y, abs_z)

        # 2. Setup Convergence Plane
        d_min, d_max = depth_float.min(), depth_float.max()
        z_min = (abs_extrusion_point[2] * extrusion_scale) * d_min
        z_max = (abs_extrusion_point[2] * extrusion_scale) * d_max

        # Convergence: 0 = z_max (Foreground), 1 = z_min (Background)
        z_conv = z_max * (1.0 - convergence) + z_min * convergence
        dist_to_conv = camera_distance - z_conv

        # 3. Setup Uniforms
        self.prog["img_width"].value = float(self.width)
        self.prog["img_height"].value = float(self.height)
        self.prog["extrusion_point"].value = tuple(float(x) for x in abs_extrusion_point)
        self.prog["extrusion_scale"].value = float(extrusion_scale)
        self.prog["tex_color"].value = 0
        self.prog["tex_depth"].value = 1

        self.tex_color.use(0)
        self.tex_depth.use(1)

        Near = 1.0
        Far = camera_distance + 1000.0

        # Screen bounds at the convergence plane
        screen_l, screen_r = -self.width / 2.0, self.width / 2.0
        screen_b, screen_t = -self.height / 2.0, self.height / 2.0

        # Render Function for each eye
        def render_eye(fbo, cx):
            # FIXED ZOOM: We define the "Full Frame" boundary at the physical camera_distance (Z=0).
            # This prevents the "shrinking/expanding" effect when convergence moves.
            base_l, base_r = screen_l * (Near / camera_distance), screen_r * (Near / camera_distance)
            base_b, base_t = screen_b * (Near / camera_distance), screen_t * (Near / camera_distance)

            # CONVERGENCE SHIFT (Off-Axis):
            # We offset the frustum window to align the two views at the dist_to_conv plane.
            # Shift = -cx * (Near / dist_to_conv)
            conv_shift = -cx * (Near / dist_to_conv)

            L, R = base_l + conv_shift, base_r + conv_shift
            B, T = base_b, base_t  # Vertical shift is 0 for parallel stereo

            Proj = self._perspective_off_axis(L, R, B, T, Near, Far)
            View = self._view_matrix(cx, 0.0, camera_distance)

            # CRITICAL: Matrix transpose in numpy returns a VIEW.
            # tobytes() on a view returns the original memory order, not the transposed one.
            # We must force a copy to ensure column-major layout for OpenGL.
            self.prog["Proj"].write(Proj.T.copy().tobytes())
            self.prog["View"].write(View.T.copy().tobytes())

            fbo.use()
            fbo.clear()
            self.vao.render(moderngl.TRIANGLES)

            # Read back to Numpy
            raw = fbo.read(components=3, dtype="f1")
            img = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))

            # Flip Y back to OpenCV standard and convert RGB to BGR
            img = np.ascontiguousarray(cv2.flip(img, 0))
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 4. Render Left Eye (Shift camera left)
        left_img = render_eye(self.fbo_left, cx=-disparity_world / 2.0)

        # 5. Render Right Eye (Shift camera right)
        right_img = render_eye(self.fbo_right, cx=disparity_world / 2.0)

        return left_img, right_img

    def release(self):
        """Releases the GPU memory. Important when running loops in Jupyter Notebooks."""
        self.vbo.release()
        self.ibo.release()
        self.vao.release()
        self.tex_color.release()
        self.tex_depth.release()
        self.fbo_left.release()
        self.fbo_right.release()
        self.ctx.release()


def run_fusion_stereo(
    image: np.ndarray,
    depth: np.ndarray,
    disparity: float = 25.0,
    convergence: float = 0.5,
    view_bias: float = 0.0,
    dolly_zoom: float = 0.0,
    extrusion_scale: float = 0.1,
    camera_distance: float = None,
    density_x: int = 512,
    density_y: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Single-call convenience function to process an image/depth pair.

    Args:
        image: Color image (H, W, 3).
        depth: Grayscale depth (H, W).
        disparity: 0.0 to 100.0 (Percentage of screen width * 10).
        convergence: 0.0 (BG) to 1.0 (FG).
        view_bias: -1.0 (Left Eye) to 1.0 (Right Eye).
        dolly_zoom: Z-offset relative to camera.
        extrusion_scale: How much to extrude (depth strength).
        camera_distance: Distance of camera from plane.
        density_x: Mesh divisions in X.
        density_y: Mesh divisions in Y.

    Returns:
        Left and Right eye images.
    """
    h, w = image.shape[:2]
    engine = FusionStereoReplicator(w, h, density_x=density_x, density_y=density_y)
    try:
        left, right = engine.process_frame(
            image=image,
            depth_map=depth,
            disparity=disparity,
            convergence=convergence,
            view_bias=view_bias,
            dolly_zoom=dolly_zoom,
            extrusion_scale=extrusion_scale,
            camera_distance=camera_distance,
        )
    finally:
        engine.release()
    return left, right


# ==========================================
# Example Usage / Testing Logic
# ==========================================
if __name__ == "__main__":
    # 1. Create dummy test data (Or load your own via cv2.imread)
    W, H = 1920, 1080

    # Create a simple striped image
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for x in range(0, W, 100):
        cv2.rectangle(img, (x, 0), (x + 50, H), (200, 100, 50), -1)

    # Create a gradient depth map (Left side is far=0, Right side is near=255)
    depth = np.linspace(0, 255, W, dtype=np.uint8)
    depth = np.tile(depth, (H, 1))

    # 2. Initialize our 3D replicator class
    print("Initializing GPU 3D Engine...")
    stereo_engine = FusionStereoReplicator(width=W, height=H, density_x=512, density_y=512)

    print("Rendering Frame...")
    # 3. Process the frame using specific Fusion-like parameters
    left_view, right_view = stereo_engine.process_frame(
        image=img,
        depth_map=depth,
        disparity=25.0,  # 2.5% of the screen width
        convergence=0.5,  # 0.5 = converge exactly halfway between foreground and background
        extrusion_scale=0.15,  # Foreground pushes 15% of the way to the camera
    )

    # Clean up GPU context
    stereo_engine.release()

    # 4. Save to disk to verify
    side_by_side = np.hstack((left_view, right_view))
    cv2.imwrite("true_3d_stereo_output.jpg", side_by_side)
    print("Successfully rendered and saved to 'true_3d_stereo_output.jpg'")
