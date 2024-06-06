# Sketch-to-Mesh 

## Introduction:
  This application's purpose is to convert a sketch or drawing into a mesh that can be imported into Blender. 
The application will take a set of images in .png extensions and run those images through a machine-learning model to output a usable mesh. 
This mesh will then be exported so that the user can do what they want with the file, such as copying and sharing the file or personally using it.

## Technologies:
  - bcrypt
  - open CV
  - bpy (Blender Python Library)
  - pymongo
  - tensorflow
  - pytorch

## Features: 
  Sketch-To-Mesh takes in multiple images and then outputs a mesh based on those images.
  The addon also has a database feature that allows users to save images and mesh objects to be accessed at a later date.

## Installation:
- First, the user must have Blender Installed.
- Once Blender is installed the user must download the Sketch-to-Mesh file and save it in a place they can find again
- Open Blender navigation to the Edit tab and then navigate to the Preferences button at the bottom
- Once the preference tab is open navigate to the Add-On tab
- Once in the add-On tab click the install button select the Sketch-To-Mesh Folder and press Install Add-on
- You should now see the Generic Sketch-To-Mesh add-on selected in the tab. 
- You may need to refresh the addons or restart Blender

## Development Setup:
  If you would like to further develop the sketch-to-mesh addon you first would need a good idea of how the Python language works. 
  First, you might want to make a new file for your new feature. To use this feature use this "from 'YourFeature' import 'YourClass'" 
  Next, you might want to use one of the operators already made and copy it to your file. Make your changes to the operator.
  Finally, put your operator into the register and unregister

## Required Add-ons:
  StM requires a couple of different addons to be utilized. In Blender, click on Edit->Preferences->Addons and please add any missing addons.

<img src="assets/xthWeewpng.png" width="800">

## Usage:
  To access the plugin inside Blender, first you need to find the sidebar tab and select the STM plugin. 
  For further information, please refer to our documentation [PLACEHOLDER].

  <img src="assets/KqsD0pj.png" width="800"> 
  <img src="assets/yQSLf7Q.png" width="800">
  <img src="assets/aWGsxR.png" width="800">
  

## Contributors:
  - Judah Smith-Dyer
  - Rafael Fernandes Da Silva
  - James Burns
  - Kliment Behr

## Project Status:
  Pre-alpha

## License: 
MIT License
  Copyright (c) [2024] [Sketch-To-Mesh]
  
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:
  
  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.
  
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.


