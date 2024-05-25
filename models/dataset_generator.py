import bpy
import os
import math
import random
import uuid

# Set up the output directory
output_base_dir = r'C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\datasets'
os.makedirs(output_base_dir, exist_ok=True)

# Clear the scene before the next render
def clear_scene():
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

# Edit a cone to look like a pyramid
def add_pyramid(size=1, location=(0, 0, 0)):
    bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=size, depth=size, location=location)

#images_per_size = 3  # in reality this is the amount of sizes.
amount_of_renders_angle = 40  # in reality this is the amount of sizes.

# List of objects to create
def create_geometric_objects():
    sizes = [1, 2, 3]
    shapes_info = [
        {'func': bpy.ops.mesh.primitive_cube_add, 'param': 'size', 'name': 'Cube'},
        {'func': bpy.ops.mesh.primitive_uv_sphere_add, 'param': 'radius', 'name': 'Sphere'},
        {'func': bpy.ops.mesh.primitive_cone_add, 'param': 'radius1', 'name': 'Cone'},
        {'func': bpy.ops.mesh.primitive_cylinder_add, 'param': 'radius', 'name': 'Cylinder'},
        {'func': add_pyramid, 'param': 'size', 'name': 'Pyramid'}
    ]
    
    for shape_info in shapes_info:
        num_images = amount_of_renders_angle * len(sizes)
        print(f"Generating {num_images} images for {shape_info['name']}...")
        counter = 0
        for size in sizes:
            clear_scene()
            kwargs = {shape_info['param']: size, 'location': (0, 0, 0)}
            shape_info['func'](**kwargs)
            obj = bpy.context.active_object
            obj.name = f"{shape_info['name']}_{size}_{counter}"
            render_object(obj, shape_info['name'], size, counter)
            bpy.data.objects.remove(obj)  # Important: delete the object after rendering
            print(f"Generated {shape_info['name']} image {counter+1} of size {size}")
            counter += 1

# Camera and rendering settings
scene = bpy.context.scene
scene.render.resolution_x = 1080
scene.render.resolution_y = 1080
scene.render.resolution_percentage = 100
scene.render.engine = 'CYCLES'

# Add a single light source
def add_light():
    light_data = bpy.data.lights.new(name="Light", type='POINT')
    light_object = bpy.data.objects.new(name="Light", object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = (5, 5, 5)
    light_data.energy = 1000

add_light()

# Enable Freestyle for outlines
scene.render.use_freestyle = True

# Configure Freestyle settings
view_layer = bpy.context.view_layer
freestyle = view_layer.freestyle_settings
line_set = freestyle.linesets.new(name="LineSet")
line_set.select_silhouette = True
line_set.select_border = True
line_set.select_crease = True
line_set.select_contour = True
line_set.select_edge_mark = True
line_style = line_set.linestyle
line_style.thickness = 1.0
line_style.color = (0, 0, 0)

cam = scene.camera

# Generate unique random camera angles
def generate_unique_random_angles(num_angles=40):  # should be 40
    angles = set()
    while len(angles) < num_angles:
        angle = (math.radians(random.uniform(0, 360)), math.radians(random.uniform(-90, 90)))
        if angle not in angles:
            angles.add(angle)
    return list(angles)

# Points camera to object
def point_camera_to_object(obj):
    direction = obj.location - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()

# Rendering function
def render_object(obj, shape_name, size, index):
    angles = generate_unique_random_angles(num_angles=40)  # Adjust the number of angles here
    for angle in angles:
        cam.location.x = obj.location.x + 10 * math.cos(angle[1]) * math.cos(angle[0])
        cam.location.y = obj.location.y + 10 * math.cos(angle[1]) * math.sin(angle[0])
        cam.location.z = obj.location.z + 10 * math.sin(angle[1])
        point_camera_to_object(obj)

        # Transparent object
        mat = bpy.data.materials.new(name="TransparentMat")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs['Base Color'].default_value = (0, 0, 0, 0)  # Transparent color
        bsdf.inputs['Alpha'].default_value = 0
        
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        # Set background to white
        scene.render.film_transparent = False
        world = bpy.context.scene.world
        world.use_nodes = True
        bg = world.node_tree.nodes['Background']
        bg.inputs[0].default_value = (1, 1, 1, 1)

        # Generate a unique filename using uuid
        unique_id = uuid.uuid4()
        filename = f"{shape_name}_{size}_{index}_{unique_id}_outline_{math.degrees(angle[0]):.2f}_{math.degrees(angle[1]):.2f}.png"
        scene.render.filepath = os.path.join(output_base_dir, filename)
        bpy.ops.render.render(write_still=True)

        print(f"Rendered {filename}")
        index = index + 1

# Clear the scene and create objects
clear_scene()
create_geometric_objects()
