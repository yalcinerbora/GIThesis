# GIThesis
My MSc. Thesis about GI.

It has a sample implementation of a voxel cone tracing GI implementation[[1][4]] and a simple deferred renderer. It provides an example of transforming voxels (instead of voxelizing the mesh) to support dynamic meshes (i.e. meshes with skeletal transformations). Paper Link: [[2][6]]

This implementation is old (it was written ~9 years ago) and tried to achieve minimal driver overhead (for the deferred rendering part only). The idea comes from this [presentation][5]:

 - Each shadow map pass is a single `glMultiDrawElementsIndirect` call (one per light). This single draw call writes to multiple framebuffers (cubemap faces for point lights and directional lights). Each cube face represents a cascade for directional lights.
 - Depth prepass is a single `glMultiDrawElementsIndirect` drawcall as well.
 - Actual GBuffer write is unfortunately one draw call per mesh. Single draw call can be achieved with hardware specific extensions (i.e. bindless textures) but I did not implemented it.
 - Light accumulation pass (i.e. shading using the GBuffer) is single draw call as well. In this case each point light "renders" a sphere (and this sphere is instanced per point light) and each directional light renders a full screen quad. Shadow maps are queried for shadowing as usual.

 You can check the [DeferredRenderer.cpp][7] for implementation.

 Actual voxel transformation and Sparse Voxel Octree (SVO) construction is implemented in CUDA. I did not bother implementing these portions in OpenGL since utilizing atomic operations and managing trees without pointers was tedious (at that time this is implemented around 2016). I would argue that it still is (maybe Slang would change this we will see).

## Dependencies

Requires CUDA 12.6. Project is Windows only and comes with VS project files. It should work on older versions of CUDA, but you need to edit the "Build\Windows\GIThesis\GIThesis.vcxproj" file. Search for "CUDA 12.6.props" and "CUDA 12.6.targets" and change the version according to your CUDA installation.

Everything else are provided by the repository.

## Working Directory

Working Directory can be found [here][1]

## Building

Project is windows only and has a Visual Studio Solution. A recent visual studio (hence MSVC) is required. Open the solution and build the GIThesis project. It does not have any arguments, you can change main.cpp file to load different scenes.

It utilizes a simple file format that I've designed called [GPU-friendly graphics (GFG)][3]. A compiled version of that library is in repository. It can only load that file format. For more information please follow the link.

## Usage

Please download the contents of the working directory file to the "WorkingDir" folder, and run the main executable.
You should see Crytek's Sponza scene with a walking character. Use numpad 7-8 to change between GI solution and no GI (deferred renderer only) version. Use numpad 1-3 to change between FPS-like and Modelling software-like (i.e. Maya) movement schemes. (Press question mark on the top left corner of the screen for more info)

GUI has some performance indicators, as well as some debug views. Press the "Render" combo box to check the debug views. On some of these views numpad `+` `-` and numpad `*` `/` may change the views.

## License

GIThesis related code is released under the MIT License. See [LICENSE][2] for details.


[1]: https://drive.google.com/file/d/1ziLFsdBNc6R4EPuRx0zPS0SI4Vqj75m7/view?usp=sharing
[2]: https://github.com/yalcinerbora/GIThesis/blob/master/LICENSE
[3]: https://github.com/yalcinerbora/GFGFileFormat
[4]: https://www.icare3d.org/research/GTC2012_Voxelization_public.pdf
[5]: https://gdcvault.com/play/1020791/Approaching-Zero-Driver-Overhead-in
[6]: https://user.ceng.metu.edu.tr/~ys/pubs/vt-jrtip19.pdf
[7]: Source/DeferredRenderer.cpp
