"""Synthetic camera paths used to render fly-through views."""

import numpy as np
import torch


def _translate_z(t):
    return np.asarray(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def _rotate_x(phi):
    return np.asarray(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def _rotate_y(th):
    return np.asarray(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def pose_spherical(theta, phi, radius):
    c2w = _translate_z(radius)
    c2w = _rotate_x(phi / 180.0 * np.pi) @ c2w
    c2w = _rotate_y(theta / 180.0 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def radius_func(angle, a, b):
    theta = (angle - (36 - 180)) * np.pi / 180
    return a * b / np.sqrt(a * a * np.sin(theta) ** 2 + b * b * np.cos(theta) ** 2)


def get_render_poses(scene="Barn"):
    stride = 120
    parameters = {
        "Ignatius": [1.7, 1.7, -87.0],
        "Truck": [2.5, 1.5, 91.0],
        "Caterpillar": [2.2, 2.2, -89.0],
        "Family": [0.9, 0.9, -91.0],
        "Barn": [2.5, 2.5, 88.0],
        "Character": [1.2, 1.2, -105.0],
        "Fountain": [1.2, 1.2, -105.0],
        "Jade": [1.2, 1.2, -105.0],
        "maneki": [1, -1, -30],
        "lego": [1, 4, -30],
    }
    factors = {
        "Ignatius": 1.0,
        "Truck": 1.0,
        "Caterpillar": 25.0,
        "Family": 35.0,
        "Barn": 1.0,
        "Character": 1.0,
        "Fountain": 1.0,
        "Jade": 1.0,
        "maneki": 40.0,
        "lego": 10.0,
    }
    # tanks_and_temples.yml ships scene_1.index "truck" while this table is keyed
    # "Truck", so match without regard to case, and say which scenes are covered
    # instead of raising a bare KeyError for one that is not.
    by_lower = {name.lower(): name for name in parameters}
    if scene.lower() not in by_lower:
        raise ValueError(
            "No fly-through camera path is defined for scene '{}'. "
            "--render_frame_type onfly covers only: {}. Use "
            "--render_frame_type all, custom or range for any other scene.".format(
                scene, ", ".join(sorted(parameters))
            )
        )
    scene = by_lower[scene.lower()]

    a, b, phi = parameters[scene]
    a *= factors[scene]
    b *= factors[scene]

    if scene == "maneki":
        stride = 20
        render_poses = np.stack(
            [
                pose_spherical(90, angle, 40)
                for angle in np.linspace(-70, -100, stride + 1)[:-1]
            ],
            0,
        )
    elif scene == "lego":
        render_poses = np.stack(
            [
                pose_spherical(angle, phi, b)
                for angle in np.linspace(-180, 180, stride + 1)
            ],
            0,
        )
    else:
        render_poses = np.stack(
            [
                pose_spherical(angle, phi, radius_func(angle, a, b))
                for angle in np.linspace(-180, 180, stride + 1)[:-1]
            ],
            0,
        )

    return torch.tensor(render_poses, dtype=torch.float32)
