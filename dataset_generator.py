import bpy
import os
import math
import random

output_dir = r'C:\Users\RAFAEL MUITO ZIKA\Desktop\Test Images\datsets'
os.makedirs(output_dir, exist_ok=True)

# i clear everything from the scene beforehand. when you open blender there is a default cube
def clear_scene():
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

# list of objects i want to be creating.
def create_geometric_objects():
    sizes = [1, 2, 3]
    shapes_info = [
        {'func': bpy.ops.mesh.primitive_cube_add, 'param': 'size'},
        {'func': bpy.ops.mesh.primitive_uv_sphere_add, 'param': 'radius'},
        {'func': bpy.ops.mesh.primitive_cone_add, 'param': 'radius1'},
        {'func': bpy.ops.mesh.primitive_cylinder_add, 'param': 'radius'},
        {'func': bpy.ops.mesh.primitive_pyramid_add, 'param': 'size'}
    ]
    
    for shape_info in shapes_info:
        for size in sizes:
            kwargs = {shape_info['param']: size, 'enter_editmode': False, 'location': (0, 0, 0)}
            shape_info['func'](**kwargs)
            obj = bpy.context.active_object
            obj.name = f"{obj.type}_{size}"
            render_object(obj)
            bpy.data.objects.remove(obj)  # this is simportant. after i render i need to delete the object so i add another one.

# camera and rendering settings
scene = bpy.context.scene
scene.render.resolution_x = 1080
scene.render.resolution_y = 1080
scene.render.resolution_percentage = 100
scene.render.engine = 'CYCLES'

cam = scene.camera
light_settings = [('POINT', 1000), ('SUN', 1000)]

# generate camera angles
# this is to create variance between the images. had to research a bit
def generate_camera_angles():
    return [(math.radians(i*360/10), math.radians(30)) for i in range(10)]

camera_angles = generate_camera_angles()

# function to point camera to object
# found this somewhere in the internet but i need the camera to be looking directly at the cube. maybe i will add some variance to this, like different
# camera positions? i am currently only kinda rotating the camera, but not exploring the x and y 
def point_camera_to_object(obj):
    direction = obj.location - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler() # what the fuck is this

# rendering function 
# basically saving the name of the  object, the angle of the camera, the type of light variance and itensity
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
            
            filename = f"{obj.name}_angle_{math.degrees(angle[0]):.2f}_{math.degrees(angle[1]):.2f}_{light_type}_{intensity}.png"
            scene.render.filepath = os.path.join(output_dir, filename)
            bpy.ops.render.render(write_still=True)
            
            bpy.data.objects.remove(light_object)
            bpy.data.lights.remove(light_data)

clear_scene()
create_geometric_objects()
