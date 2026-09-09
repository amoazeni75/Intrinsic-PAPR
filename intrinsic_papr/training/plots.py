"""The two figures saved during training: the image panel and the point cloud."""

import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# The six panels each image group shows, in drawing order, with the words that
# go into the panel title around the image kind ("rgb" or "albedo").
_PANELS = (
    ("train_tgt", "tr tgt {}"),
    ("train_tgt_patch", "tr tgt {} patch"),
    ("train_pred_patch", "tr pred {} patch"),
    ("test_tgt", "eval tgt {}"),
    ("test_pred", "eval pred {}"),
    ("test_pred_foreground", "eval pred foreground {}"),
)

# The point-cloud figure shows the same scatter from four angles.
_PCD_VIEWS = (
    (0.0, 90, "Point Cloud View 1"),
    (0.0, 180, "Point Cloud View 2"),
    (0.0, 270, "Point Cloud View 3"),
    (89.9, 90, "Point Cloud View 1 Up"),
)

_PANEL_TITLE_PAD = 0


def get_colors(weights):
    num_points = weights.shape[0]
    weights = (weights - weights.min()) / (weights.max() - weights.min())
    colors = np.full((num_points, 3), [1.0, 0.0, 0.0])
    colors[:, 0] *= weights[:num_points]
    colors[:, 2] = 1 - weights[:num_points]
    return colors


def _draw_panels(fig, rows, cols, plot_index, step, panels, kind, suffix=""):
    """Draw one group's six panels and return the index the next panel takes."""
    for name, title in _PANELS:
        plot_index += 1
        ax = fig.add_subplot(rows, cols, plot_index)
        ax.imshow(panels[name])
        ax.set_title(
            "Iter: {} {}{}".format(step, title.format(kind), suffix),
            pad=_PANEL_TITLE_PAD,
        )
    return plot_index


def _draw_point_cloud(ax, points_np, pt_plot_scale, color, size=None):
    """Set up a 3D axis over the scene's extent and scatter the points into it."""
    ax.set_xlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_ylim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_zlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # this matplotlib's Axes3D.scatter rejects s=None, so only pass a size when set
    size_kwargs = {} if size is None else {"s": size}
    ax.scatter(
        points_np[:, 0], points_np[:, 1], points_np[:, 2], c=color, **size_kwargs
    )


def _figure_to_image(fig):
    buffer = io.BytesIO()
    fig.canvas.print_png(buffer)
    img = Image.open(buffer)
    plt.close()
    return img


def get_training_main_plot(
    index,
    step,
    render_panels,
    depth_np,
    points_np,
    pt_plot_scale,
    bg_attentions,
    bg_masks,
    render_raw_panels=None,
    albedo_panels=None,
    albedo_raw_panels=None,
    points_conf_scores_np=None,
):
    """The per-eval overview figure: every image group, then depth, points and background.

    Each ``*_panels`` argument is the six-entry mapping `panel_arrays` returns,
    or None when that group is not being shown.
    """
    col_counts = 6
    col_size = 30
    row_size = 30

    # One row per active group. The albedo case starts from two rather than
    # three, which is how this has always sized itself.
    row_counts = 2 if albedo_panels is not None else 3
    if render_raw_panels is not None:
        row_counts += 1
    if albedo_panels is not None:
        row_counts += 1
    if albedo_raw_panels is not None:
        row_counts += 1
    row_size = (row_counts / 3.0) * row_size

    fig = plt.figure(figsize=(col_size, row_size))
    fig.subplots_adjust(wspace=0.2, hspace=0.2)

    plot_index = 0
    plot_index = _draw_panels(
        fig, row_counts, col_counts, plot_index, step, render_panels, "rgb"
    )
    if render_raw_panels is not None:
        plot_index = _draw_panels(
            fig,
            row_counts,
            col_counts,
            plot_index,
            step,
            render_raw_panels,
            "rgb",
            " raw space",
        )
    if albedo_panels is not None:
        plot_index = _draw_panels(
            fig, row_counts, col_counts, plot_index, step, albedo_panels, "albedo"
        )
        if albedo_raw_panels is not None:
            plot_index = _draw_panels(
                fig,
                row_counts,
                col_counts,
                plot_index,
                step,
                albedo_raw_panels,
                "albedo",
                " raw space",
            )

    plot_index += 1
    ax = fig.add_subplot(row_counts, col_counts, plot_index)
    depth_image = ax.imshow(depth_np)
    fig.colorbar(depth_image, ax=ax)
    ax.set_title("depth map", pad=_PANEL_TITLE_PAD)

    plot_index += 1
    ax = fig.add_subplot(row_counts, col_counts, plot_index, projection="3d")
    _draw_point_cloud(
        ax,
        points_np,
        pt_plot_scale,
        "grey" if points_conf_scores_np is None else get_colors(points_conf_scores_np),
    )
    ax.set_title("Point Cloud", pad=_PANEL_TITLE_PAD)

    for image, title in ((bg_attentions, "bg attentions"), (bg_masks, "bg masks")):
        plot_index += 1
        ax = fig.add_subplot(row_counts, col_counts, plot_index)
        ax.imshow(image)
        ax.set_title(title, pad=_PANEL_TITLE_PAD)

    fig.suptitle(
        "Main Plot\n%s\niter %d\nnum pts: %d" % (index, step, points_np.shape[0])
    )
    return _figure_to_image(fig)


def get_training_pcd_plot(
    index,
    step,
    ro,
    rd,
    points_np,
    coord_scale,
    pt_plot_scale,
    points_conf_scores_np=None,
):
    """The point cloud from four angles, with the eval camera ray drawn in.

    With confidence scores there are two extra panels showing their spread.
    """
    num_plots = 6 if points_conf_scores_np is not None else 4
    fig = plt.figure(figsize=(5 * num_plots, 6))

    H, W, _ = rd.shape
    color = (
        "orange" if points_conf_scores_np is None else get_colors(points_conf_scores_np)
    )
    center_ray = rd[H // 2, W // 2]

    for plot_index, (elev, azim, title) in enumerate(_PCD_VIEWS, start=1):
        ax = fig.add_subplot(1, num_plots, plot_index, projection="3d")
        ax.view_init(elev=elev, azim=azim)
        _draw_point_cloud(ax, points_np, pt_plot_scale, color, 0.8 * coord_scale)
        ax.scatter(ro[0], ro[1], ro[2], c="red", s=10)
        ax.quiver(
            ro[0],
            ro[1],
            ro[2],
            center_ray[0],
            center_ray[1],
            center_ray[2],
            length=2,
            alpha=1,
            color="blue",
        )
        ax.set_title(title)

    if points_conf_scores_np is not None:
        ax = fig.add_subplot(1, num_plots, 5)
        ax.scatter(range(len(points_conf_scores_np)), points_conf_scores_np)
        ax.set_title("Confidence Scores scatter plot")

        ax = fig.add_subplot(1, num_plots, 6)
        ax.hist(points_conf_scores_np, bins=np.linspace(-1, 1, 100).tolist())
        ax.set_title("Confidence Scores histogram")

    fig.suptitle("Point Clouds\n%s\niter %d" % (index, step))
    return _figure_to_image(fig)
