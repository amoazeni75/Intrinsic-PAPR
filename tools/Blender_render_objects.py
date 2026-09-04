import argparse
import json
import os

import bpy
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar


def get_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description="Renders given obj file by rotating a camera around it."
    )
    parser.add_argument(
        "--n_views",
        type=int,
        default=3,
        help="The number of views to be rendered.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=800,
        help="The resolution (in pixels) of the rendered images.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="output",
        help="The path to the directory where the results will be stored.",
    )
    parser.add_argument(
        "--blend_file_path",
        type=str,
        default=None,
        help="The path to the .blend file to be rendered.",
    )
    parser.add_argument(
        "--frame_code",
        type=str,
        default="",
        help="The code to be added to the end of the frame name.",
    )
    parser.add_argument("--save_format", choices=["exr", "png"], default="png")
    parser.add_argument(
        "--load_transformation_matrix",
        action="store_true",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--transformation_matrix_path", type=str, default=None)
    return parser.parse_args()


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


def load_blend_file(filepath):
    """Load a .blend file"""
    bpy.ops.wm.open_mainfile(filepath=filepath)


def load_transformation_matrix(filepath):
    with open(filepath, "r") as fp:
        meta = json.load(fp)
    poses = []
    for i, frame in enumerate(meta["frames"]):
        poses.append(np.array(frame["transform_matrix"]))
    return poses


def find_full_filename(directory, partial_filename):
    # Traverse the directory to find files starting with the partial filename
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(partial_filename):
                # Return the full path of the file
                return os.path.join(root, file)
    return None


args = get_args()

# Load the .blend file
if args.blend_file_path is not None:
    load_blend_file(args.blend_file_path)

if args.load_transformation_matrix:
    views = load_transformation_matrix(args.transformation_matrix_path)
    RANDOM_VIEWS = False
else:
    views = None
    RANDOM_VIEWS = True
    UPPER_VIEWS = True


# Configuration
VIEWS = args.n_views
RESOLUTION = args.resolution
RESULTS_PATH = args.results_path
if args.save_format == "png":
    COLOR_DEPTH = 8
    FORMAT = "PNG"
elif args.save_format == "exr":
    COLOR_DEPTH = 16
    FORMAT = "OPEN_EXR"


fp = os.path.join(RESULTS_PATH, args.split)
albedo_fp = os.path.join(RESULTS_PATH, f"{args.split}_albedo_GT")
normal_fp = os.path.join(RESULTS_PATH, f"{args.split}_normal_GT")
if not os.path.exists(fp):
    os.makedirs(fp)
    print("Created directory: " + fp)
if not os.path.exists(albedo_fp):
    os.makedirs(albedo_fp)
    print("Created directory: " + albedo_fp)

if not os.path.exists(normal_fp):
    os.makedirs(normal_fp)
    print("Created directory: " + normal_fp)

out_data = {"camera_angle_x": bpy.data.objects["Camera"].data.angle_x, "frames": []}

# Set render settings
scene = bpy.context.scene
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = FORMAT
scene.render.image_settings.color_depth = str(COLOR_DEPTH)
scene.render.film_transparent = True

# check if "RenderLayer" key exists
if "RenderLayer" in scene.view_layers:
    scene.view_layers["RenderLayer"].use_pass_diffuse_color = True
    scene.view_layers["RenderLayer"].use_pass_diffuse_direct = True
    scene.view_layers["RenderLayer"].use_pass_diffuse_indirect = True
    scene.view_layers["RenderLayer"].use_pass_normal = True

# Set up rendering nodes
scene.use_nodes = True
tree = scene.node_tree
links = tree.links

# Clear default nodes
for node in tree.nodes:
    tree.nodes.remove(node)

# Create Render Layers node
render_layers = tree.nodes.new("CompositorNodeRLayers")

# Enable Albedo Pass
if "View Layer" in scene.view_layers:
    scene.view_layers["View Layer"].use_pass_diffuse_color = True
    scene.view_layers["View Layer"].use_pass_diffuse_direct = True
    scene.view_layers["View Layer"].use_pass_diffuse_indirect = True
    scene.view_layers["View Layer"].use_pass_normal = True

# Create node to combine albedo with alpha
mix_node = tree.nodes.new(type="CompositorNodeMixRGB")
mix_node.blend_type = "MULTIPLY"
links.new(render_layers.outputs["DiffCol"], mix_node.inputs[1])
links.new(render_layers.outputs["Alpha"], mix_node.inputs[2])

# Create Set Alpha node for albedo
set_alpha_albedo = tree.nodes.new(type="CompositorNodeSetAlpha")
links.new(mix_node.outputs[0], set_alpha_albedo.inputs[0])
links.new(render_layers.outputs["Alpha"], set_alpha_albedo.inputs[1])

# Create Albedo Output node
albedo_output = tree.nodes.new(type="CompositorNodeOutputFile")
albedo_output.label = "Albedo Output"
albedo_output.base_path = albedo_fp
albedo_output.format.file_format = FORMAT
albedo_output.format.color_mode = "RGBA"
links.new(set_alpha_albedo.outputs[0], albedo_output.inputs[0])

# Create nodes to combine diffuse direct and indirect
add_node = tree.nodes.new(type="CompositorNodeMixRGB")
add_node.blend_type = "ADD"
links.new(render_layers.outputs["DiffDir"], add_node.inputs[1])  # Diffuse Direct
links.new(render_layers.outputs["DiffInd"], add_node.inputs[2])  # Diffuse Indirect

# Create Set Alpha node for normal
set_alpha_normal = tree.nodes.new(type="CompositorNodeSetAlpha")
links.new(render_layers.outputs["Normal"], set_alpha_normal.inputs[0])
links.new(render_layers.outputs["Alpha"], set_alpha_normal.inputs[1])

# Normal
normal_output = tree.nodes.new(type="CompositorNodeOutputFile")
normal_output.label = "Normal Output"
normal_output.base_path = normal_fp
normal_output.format.file_format = FORMAT
normal_output.format.color_mode = "RGBA"
links.new(set_alpha_normal.outputs[0], normal_output.inputs[0])

# Ensure there is a camera in the scene
if "Camera" not in scene.objects:
    raise Exception("No camera found in the scene.")

# Camera setup
cam = scene.objects["Camera"]
cam.location = (0, 4.0, 0.5)
cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"

# Parent camera to an empty object at the origin
b_empty = bpy.data.objects.new("Empty", None)
b_empty.location = (0, 0, 0)
cam.parent = b_empty
scene.collection.objects.link(b_empty)
bpy.context.view_layer.objects.active = b_empty


# Rendering loop with progress bar
if views is not None:
    pbar = tqdm(views, desc="Rendering Views")
else:
    pbar = tqdm(range(VIEWS), desc="Rendering Views")
for i, view in enumerate(pbar):
    filename = f"r_{i}.{args.save_format}"
    scene.render.filepath = os.path.join(fp, filename)
    albedo_output.file_slots[0].path = filename
    normal_output.file_slots[0].path = filename

    if RANDOM_VIEWS:
        if UPPER_VIEWS:
            rot = np.random.uniform(0, 1, size=3) * (1, 0, 2 * np.pi)
            rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi / 2)
        else:
            rot = np.random.uniform(0, 2 * np.pi, size=3)
        b_empty.rotation_euler = rot
    elif args.load_transformation_matrix:
        # we need to set the camera and render the object given a numpy 4x4 transformation matrix
        b_empty.matrix_world = bpy.context.scene.objects["Camera"].matrix_world.copy()
        b_empty.matrix_world = view.T
        bpy.context.scene.objects["Camera"].matrix_world = view.T
    else:
        b_empty.rotation_euler[2] += 2 * np.pi / VIEWS

    bpy.ops.render.render(write_still=True)

    # Save frame data
    frame_data = {
        "file_path": scene.render.filepath,
        "rotation": list(b_empty.rotation_euler),
        "transform_matrix": listify_matrix(cam.matrix_world),
    }
    out_data["frames"].append(frame_data)

    full_name_albedo = find_full_filename(
        directory=albedo_fp, partial_filename=filename
    )
    os.rename(full_name_albedo, os.path.join(albedo_fp, filename))

    full_name_normal = find_full_filename(
        directory=normal_fp, partial_filename=filename
    )
    os.rename(full_name_normal, os.path.join(normal_fp, filename))

# Save JSON data
if not args.load_transformation_matrix:
    with open(fp + "/transforms.json", "w") as out_file:
        json.dump(out_data, out_file, indent=4)

print("Rendering completed.")
