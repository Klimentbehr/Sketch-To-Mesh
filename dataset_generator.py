import bpy
import os
import math
import random

# Output directories
output_base_dir = r'C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\datasets'
train_dir = os.path.join(output_base_dir, 'train')
validation_dir = os.path.join(output_base_dir, 'validation')
test_dir = os.path.join(output_base_dir, 'test')

# Ensure directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Clear everything from the scene beforehand
def clear_scene():
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

# Function to add a pyramid
def add_pyramid(size=1, location=(0, 0, 0)):
    bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=size, depth=size, location=location)

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
        for size in sizes:
            kwargs = {shape_info['param']: size, 'location': (0, 0, 0)}
            shape_info['func'](**kwargs)
            obj = bpy.context.active_object
            obj.name = f"{shape_info['name']}_{size}"
            render_object(obj)
            bpy.data.objects.remove(obj)  # Important: delete the object after rendering

# Camera and rendering settings
scene = bpy.context.scene
scene.render.resolution_x = 1080
scene.render.resolution_y = 1080
scene.render.resolution_percentage = 100
scene.render.engine = 'CYCLES'

cam = scene.camera
light_settings = [('POINT', 1000), ('SUN', 1000)]

# Generate camera angles
def generate_camera_angles():
    return [(math.radians(i * 360 / 10), math.radians(30)) for i in range(10)]

camera_angles = generate_camera_angles()

# Point camera to object
def point_camera_to_object(obj):
    direction = obj.location - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()

# Rendering function
def render_object(obj):
    for angle in camera_angles:
        cam.location.x = obj.location.x + 10 * math.cos(angle[1]) * math.cos(angle[0])
        cam.location.y = obj.location.y + 10 * math.cos(angle[1]) * math.sin(angle[0])
        cam.location.z = obj.location.z + 10 * math.sin(angle[1])
        point_camera_to_object(obj)
        
        for light_type, intensity in light_settings:
            light_data = bpy.data.lights.new(name="New_Light", type=light_type)
            light_object = bpy.data.objects.new(name="New_Light", object_data=light_data)
            scene.collection.objects.link(light_object)
            light_object.location = (5, 5, 5)
            light_data.energy = intensity

            # Determine which dataset split to use
            split = random.choices(['train', 'validation', 'test'], [0.8, 0.1, 0.1])[0]
            output_dir = os.path.join(output_base_dir, split)
            filename = f"{obj.name}_angle_{math.degrees(angle[0]):.2f}_{math.degrees(angle[1]):.2f}_{light_type}_{intensity}.png"
            scene.render.filepath = os.path.join(output_dir, filename)
            bpy.ops.render.render(write_still=True)
            
            bpy.data.objects.remove(light_object)
            bpy.data.lights.remove(light_data)

# Clear the scene and create objects
clear_scene()
create_geometric_objects()
