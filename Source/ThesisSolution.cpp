#include "ThesisSolution.h"
#include "Globals.h"
#include "SceneI.h"
#include "DrawBuffer.h"
#include "Macros.h"
#include "Camera.h"
#include "DeferredRenderer.h"
#include "SceneLights.h"
#include "IEUtility/IEMath.h"
#include <cuda_gl_interop.h>
#include "OGLTimer.h"
#include "IEUtility/IETimer.h"
#include "GFGLoader.h"
#include "MeshBatchSkeletal.h"

const size_t ThesisSolution::InitialObjectGridSize = 256;
const float ThesisSolution::CascadeSpan = 0.6f;
const uint32_t ThesisSolution::CascadeDim = 512;

const TwEnumVal ThesisSolution::renderSchemeVals[] = 
{ 
	{ GI_DEFERRED, "Deferred" }, 
	{ GI_LIGHT_INTENSITY, "LI Buffer Only" },
	{ GI_SVO_DEFERRED, "Render SVO Deferred" },
	{ GI_SVO_LEVELS, "Render SVO Levels"},
	{ GI_VOXEL_PAGE, "Render Voxel Page" },
	{ GI_VOXEL_CACHE2048, "Render Voxel Cache 2048" },
	{ GI_VOXEL_CACHE1024, "Render Voxel Cache 1024" },
	{ GI_VOXEL_CACHE512, "Render Voxel Cache 512" }
};

AOBar::AOBar()
 : angleDegree(30.0f)
 , sampleFactor(1.0f)
 , maxDistance(250.0f)
 , intensity(1.15f)
 , bar(nullptr)
 , hidden(true)
{
	bar = TwNewBar("AOBar");
	TwDefine(" AOBar visible = false ");
	TwAddVarRW(bar, "cAngle", TW_TYPE_FLOAT, &angleDegree,
			   " label='Cone Angle' help='Cone Angle' "
			   " min=1.0 max=90.0 step= 0.01 ");
	TwAddVarRW(bar, "sFactor", TW_TYPE_FLOAT, &sampleFactor,
			   " label='Sample Factor' help='Adjusts Sampling Rate' "
			   " min=0.5 max=10.0 step=0.01 ");
	TwAddVarRW(bar, "maxDist", TW_TYPE_FLOAT, &maxDistance,
			   " label='Max Distance' help='Maximum Cone Trace Distance' "
			   " min=10.0 max=300.0 step=0.1 ");
	TwAddVarRW(bar, "intensity", TW_TYPE_FLOAT, &intensity,
			   " label='Intensity' help='Occlusion Intensity' "
			   " min=0.5 max=5.0 step=0.01 ");
	TwDefine(" AOBar valueswidth=fit ");
	TwDefine(" AOBar position='20 500' ");
	TwDefine(" AOBar size='220 100' ");
}

void AOBar::HideBar(bool hide)
{
	if(hide != hidden)
	{
		if(hide)
			TwDefine(" AOBar visible=false ");
		else
			TwDefine(" AOBar visible=true ");
		hidden = hide;
	}
}

AOBar::~AOBar()
{
	if(bar) TwDeleteBar(bar);
	bar = nullptr;
}

ThesisSolution::ThesisSolution(DeferredRenderer& dRenderer, const IEVector3& intialCamPos)
	: vertexDebugVoxel(ShaderType::VERTEX, "Shaders/VoxRender.vert")
	, vertexDebugVoxelSkeletal(ShaderType::VERTEX, "Shaders/VoxRenderSkeletal.vert")
	, vertexDebugWorldVoxel(ShaderType::VERTEX, "Shaders/VoxRenderWorld.vert")
	, fragmentDebugVoxel(ShaderType::FRAGMENT, "Shaders/VoxRender.frag")
	, bar(nullptr)
	, renderScheme(GI_VOXEL_PAGE)
	//, renderScheme(GI_VOXEL_CACHE2048)
	, gridInfoBuffer(1)
	, voxelNormPosBuffer(512)
	, voxelColorBuffer(512)
	, voxelOctree()
	, traceType(0)
	, EmptyGISolution(dRenderer)
{
	renderType = TwDefineEnum("RenderType", renderSchemeVals, GI_END);
	gridInfoBuffer.AddData({});
	for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
	{
		voxelScenes.emplace_back(intialCamPos, CascadeSpan * (0x1 << i), CascadeDim);
	}
}

ThesisSolution::~ThesisSolution()
{}

bool ThesisSolution::IsCurrentScene(SceneI& scene)
{
	return &scene == currentScene;
}

void ThesisSolution::Init(SceneI& s)
{
	// Reset GICudaScene
	voxelCaches.clear();
	for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
	{
		voxelScenes[i].Reset();
		voxelCaches.emplace_back();
		voxelCaches.back().span = CascadeSpan * (1 << i);
		voxelCaches.back().depth = (GI_CASCADE_COUNT - i - 1) + static_cast<uint32_t>(IEMath::Log2F(static_cast<float>(CascadeDim)));	
		voxelCaches[i].voxOctreeCount = 0;
		voxelCaches[i].voxOctreeSize = 0;
	}
	EmptyGISolution::Init(s);

	// Voxelization
	// and Voxel Cache Creation
	double voxelTotaltime = 0.0;
	Array32<MeshBatchI*> batches = currentScene->getBatches();
	for(unsigned int i = 0; i < batches.length; i++)
	{
		LoadBatchVoxels(batches.arr[i]);
	}

	for(unsigned int i = 0; i < voxelCaches.size(); i++)
	{
		uint32_t totalCount = 0;
		double totalSize = 0.0f;
		for(unsigned int j = 0; j < voxelCaches[i].cache.size(); j++)
		{
			totalCount += voxelCaches[i].cache[j].batchVoxCacheCount;
			totalSize += voxelCaches[i].cache[j].batchVoxCacheSize;
		}
		voxelCaches[i].totalCacheCount = totalCount;
		voxelCaches[i].totalCacheSize = totalSize;
	}
	GI_LOG("Scene voxelization completed. Elapsed time %f ms", voxelTotaltime);
	
	// Voxel Page System Linking
	for(unsigned int j = 0; j < GI_CASCADE_COUNT; j++)
		LinkCacheWithVoxScene(voxelScenes[j], voxelCaches[j], 1.5f);
	
	// Allocators Link
	// Ordering is reversed svo tree needs cascades from other to inner
	std::vector<GICudaAllocator*> allocators;
	for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
		allocators.push_back(voxelScenes[GI_CASCADE_COUNT - i - 1].Allocator());

	voxelOctree.LinkAllocators(Array32<GICudaAllocator*>{allocators.data(), GI_CASCADE_COUNT},
							   currentScene->SVOTotalSize(),
							   currentScene->SVOLevelSizes());
	svoRenderLevel = voxelOctree.SVOConsts().totalDepth;

	// Memory Usage Total
	for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
	{
		GI_LOG("Voxel Sytem #%d Total Memory Usage %f MB", i,
			   static_cast<double>(voxelScenes[i].AllocatorMemoryUsage()) / 1024.0 / 1024.0);
	}
	GI_LOG("Voxel Octree Sytem Total Memory Usage %f MB",
		   static_cast<double>(voxelOctree.MemoryUsage()) / 1024.0 / 1024.0);

	// Tw Bar Creation
	bar = TwNewBar("ThesisGI");
	TwDefine(" ThesisGI refresh=0.01 ");

	// Stuff
	TwAddVarRW(bar, "rendering", renderType,
			   &renderScheme,
			   " label='Render' help='Change what to show on screen' ");
	TwAddVarRW(bar, "giOn", TW_TYPE_BOOLCPP,
			   &giOn,
			   " label='GI On' help='Cone Tracing GI On off' ");
	TwAddSeparator(bar, NULL, NULL);
	for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
	{
		std::string start("label = 'Cascade#");
		start += std::to_string(i);
		std::string endCount(" Count' group='Voxel Cache' help='Cache voxel count.' ");
		std::string endSize(" Size(MB)' group='Voxel Cache' precision=2 help='Cache voxel total size in megabytes.' ");
		TwAddVarRO(bar, (std::string("voxCache") + std::to_string(i)).c_str(), TW_TYPE_UINT32, &voxelCaches[i].totalCacheCount,
				   (start + endCount).c_str());
		TwAddVarRO(bar, (std::string("voxCacheSize") + std::to_string(i)).c_str(), TW_TYPE_DOUBLE, &voxelCaches[i].totalCacheSize,
				   (start + endSize).c_str());
	}
	TwAddSeparator(bar, NULL, NULL);
	for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
	{
		std::string start("label = 'Cascade#");
		start += std::to_string(i);
		std::string endCount(" Count' group='Voxel Octree' help='Voxel count in octree.' ");
		std::string endSize(" Size(MB)' group='Voxel Octree' precision=2 help='Octree Voxel total size in megabytes.' ");

		TwAddVarRO(bar, (std::string("voxUsed") + std::to_string(i)).c_str(), TW_TYPE_UINT32, &voxelCaches[i].voxOctreeCount,
				   (start + endCount).c_str());
		TwAddVarRO(bar, (std::string("voxUsedSize") + std::to_string(i)).c_str(), TW_TYPE_DOUBLE, &voxelCaches[i].voxOctreeSize,
				   (start + endSize).c_str());
	}
	TwAddSeparator(bar, NULL, NULL);
	TwAddVarRO(bar, "ioTime", TW_TYPE_DOUBLE, &ioTime,
			   " label='I-O Time (ms)' group='Timings' precision=2 help='Voxel Include Exclude Timing per frame.' ");
	TwAddVarRO(bar, "updateTime", TW_TYPE_DOUBLE, &transformTime,
			   " label='Update Time (ms)' group='Timings' precision=2 help='Voxel Grid Update Timing per frame.' ");
	TwAddVarRO(bar, "svoReconTime", TW_TYPE_DOUBLE, &svoTime,
			   " label='SVO Time (ms)' group='Timings' precision=2 help='SVO Reconstruct Timing per frame.' ");
	TwAddVarRO(bar, "transferTime", TW_TYPE_DOUBLE, &debugVoxTransferTime,
			   " label='Dbg Transfer Time (ms)' group='Timings' precision=2 help='Voxel Copy to OGL Timing.' ");
	TwDefine(" ThesisGI size='325 430' ");
	TwDefine(" ThesisGI valueswidth=fit ");
}

void ThesisSolution::Release()
{
	EmptyGISolution::Release();
	if(bar) TwDeleteBar(bar);
	bar = nullptr;
}

double ThesisSolution::LoadBatchVoxels(MeshBatchI* batch)
{
	IETimer t;
	t.Start();

	// Load GFG
	std::string batchVoxFile = "vox_" + batch->BatchName() + ".gfg";	
	LoadVoxel(voxelCaches, batchVoxFile.c_str(), GI_CASCADE_COUNT,
			  batch->MeshType() == VoxelObjectType::SKEL_DYNAMIC);

	t.Stop();
	// Voxel Load Complete
	GI_LOG("Loading \"%s\" complete", batchVoxFile);
	GI_LOG("\tDuration : %f ms", t.ElapsedMilliS());
	GI_LOG("------");
	return t.ElapsedMilliS();
}

bool ThesisSolution::LoadVoxel(std::vector<SceneVoxCache>& scenes,
							   const char* gfgFileName, uint32_t cascadeCount,
							   bool isSkeletal)
{
	std::ifstream stream(gfgFileName, std::ios_base::in | std::ios_base::binary);
	GFGFileReaderSTL stlFileReader(stream);
	GFGFileLoader gfgFile(&stlFileReader);

	GFGFileError e = gfgFile.ValidateAndOpen();
	assert(e == GFGFileError::OK);

	// Assertions
	const auto& header = gfgFile.Header();
	assert((header.meshes.size() - 1) == cascadeCount);

	// First mesh contains objInfos
	const auto& meshObjCount = header.meshes.back();
	assert(meshObjCount.components.size() == cascadeCount);

	uint32_t objCount = static_cast<uint32_t>(meshObjCount.headerCore.vertexCount);
	std::vector<uint8_t> objectInfoData(gfgFile.MeshVertexDataSize(cascadeCount));
	gfgFile.MeshVertexData(objectInfoData.data(), cascadeCount);

	// Determine VoxelCount
	for(uint32_t i = 0; i < cascadeCount; i++)
	{
		const auto& mesh = header.meshes[i];

		// Special case aabbmin show span count
		assert(scenes[i].span == mesh.headerCore.aabb.min[0]);
		scenes[i].cache.emplace_back(mesh.headerCore.vertexCount, objCount, isSkeletal);

		// Load to Mem
		std::vector<uint8_t> meshData(gfgFile.MeshVertexDataSize(i));
		gfgFile.MeshVertexData(meshData.data(), i);

		auto& currentCache = scenes[i].cache.back();

		// Object gridInfo
		const auto& component = meshObjCount.components[i];
		assert(component.dataType == GFGDataType::UINT32_2);
		assert(sizeof(ObjGridInfo) == GFGDataTypeByteSize[static_cast<int>(GFGDataType::UINT32_2)]);
		assert(component.internalOffset == 0);
		assert(component.logic == GFGVertexComponentLogic::POSITION);
		assert(component.stride == sizeof(ObjGridInfo));

		currentCache.objInfo.CPUData().resize(objCount);
		std::memcpy(currentCache.objInfo.CPUData().data(),
					objectInfoData.data() + component.startOffset,
					objCount * component.stride);

		// Voxel Data
		for(const auto& component : mesh.components)
		{
			if(component.logic == GFGVertexComponentLogic::POSITION)
			{
				// NormPos
				assert(component.dataType == GFGDataType::UINT32_2);
				auto& normPosVector = currentCache.voxelNormPos.CPUData();

				normPosVector.resize(mesh.headerCore.vertexCount);
				std::memcpy(normPosVector.data(), meshData.data() +
							component.startOffset,
							mesh.headerCore.vertexCount * component.stride);
			}
			else if(component.logic == GFGVertexComponentLogic::NORMAL)
			{
				// Vox Ids
				assert(component.dataType == GFGDataType::UINT32_2);
				auto& voxIdsVector = currentCache.voxelIds.CPUData();

				voxIdsVector.resize(mesh.headerCore.vertexCount);
				std::memcpy(voxIdsVector.data(), meshData.data() +
							component.startOffset,
							mesh.headerCore.vertexCount * component.stride);
			}
			else if(component.logic == GFGVertexComponentLogic::COLOR)
			{
				// Color
				assert(component.dataType == GFGDataType::UNORM8_4);
				auto& voxColorVector = currentCache.voxelRenderData.CPUData();

				voxColorVector.resize(mesh.headerCore.vertexCount);
				std::memcpy(voxColorVector.data(), meshData.data() +
							component.startOffset,
							mesh.headerCore.vertexCount * component.stride);
			}
			else if(component.logic == GFGVertexComponentLogic::WEIGHT)
			{
				// Weight
				assert(component.dataType == GFGDataType::UINT32_2);
				auto& voxWeightVector = currentCache.voxelWeightData.CPUData();

				voxWeightVector.resize(mesh.headerCore.vertexCount);
				std::memcpy(voxWeightVector.data(), meshData.data() +
							component.startOffset,
							mesh.headerCore.vertexCount * component.stride);
			}
			else
			{
				assert(false);
			}
			currentCache.voxelRenderData.SendData();
			currentCache.voxelNormPos.SendData();
			currentCache.voxelIds.SendData();
			currentCache.objInfo.SendData();
			if(isSkeletal) currentCache.voxelWeightData.SendData();
		}

		GI_LOG("\tCascade#%d Voxels : %d", i, mesh.headerCore.vertexCount);
	}

	return true;
}


void ThesisSolution::LinkCacheWithVoxScene(GICudaVoxelScene& scene, 
										   SceneVoxCache& cache,
										   float coverageRatio)
{
	// Send it to CUDA
	Array32<MeshBatchI*> batches = currentScene->getBatches();
	assert(batches.length == cache.cache.size());
	for(unsigned int i = 0; i < cache.cache.size(); i++)
	{
		scene.LinkOGL(batches.arr[i]->getDrawBuffer().getAABBBuffer().getGLBuffer(),
					  batches.arr[i]->getDrawBuffer().getModelTransformBuffer().getGLBuffer(),
					  batches.arr[i]->getDrawBuffer().getModelTransformIndexBuffer().getGLBuffer(),
					  cache.cache[i].objInfo.getGLBuffer(),
					  cache.cache[i].voxelNormPos.getGLBuffer(),
					  cache.cache[i].voxelIds.getGLBuffer(),
					  cache.cache[i].voxelRenderData.getGLBuffer(),
					  static_cast<uint32_t>(batches.arr[i]->DrawCount()),
					  cache.cache[i].batchVoxCacheCount);
	}
	// Allocate at least all of the scene voxel
	scene.AllocateWRTLinkedData(coverageRatio);
}

void ThesisSolution::LevelIncrement()
{
	svoRenderLevel++;
	svoRenderLevel = std::min(svoRenderLevel, voxelOctree.MaxLevel());
	GI_LOG("Level %d", svoRenderLevel);
}

void ThesisSolution::LevelDecrement()
{
	svoRenderLevel--;
	svoRenderLevel = std::max(svoRenderLevel, voxelOctree.MinLevel());
	GI_LOG("Level %d", svoRenderLevel);
}

void ThesisSolution::TraceTypeInc()
{
	traceType++;
	//GI_LOG("Trace Type %d", traceType % 3);
}

void ThesisSolution::TraceTypeDec()
{
	traceType--;
	//GI_LOG("Trace Type %d", traceType % 3);
}

void ThesisSolution::DebugRenderVoxelCache(const Camera& camera, 
										   SceneVoxCache& cache)
{
	//DEBUG VOXEL RENDER
	// Frame Viewport
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0,
			   static_cast<GLsizei>(camera.width),
			   static_cast<GLsizei>(camera.height));

	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

	glDisable(GL_MULTISAMPLE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDepthFunc(GL_LEQUAL);
	glDepthMask(true);
	glColorMask(true, true, true, true);

	glClear(GL_COLOR_BUFFER_BIT |
			GL_DEPTH_BUFFER_BIT);

	// Debug Voxelize Scene
	Shader::Unbind(ShaderType::GEOMETRY);
	fragmentDebugVoxel.Bind();

	cameraTransform.Bind();
	cameraTransform.Update(camera.generateTransform());

	Array32<MeshBatchI*> batches = currentScene->getBatches();
	for(unsigned int i = 0; i < cache.cache.size(); i++)
	{		
		DrawBuffer& dBuffer = batches.arr[i]->getDrawBuffer();
		dBuffer.getAABBBuffer().BindAsShaderStorageBuffer(LU_AABB);
		dBuffer.getModelTransformBuffer().BindAsShaderStorageBuffer(LU_MTRANSFORM);
		dBuffer.getModelTransformIndexBuffer().BindAsShaderStorageBuffer(LU_MTRANSFORM_INDEX);

		if(batches.arr[i]->MeshType() == VoxelObjectType::SKEL_DYNAMIC)
		{
			// Shader Vert
			vertexDebugVoxelSkeletal.Bind();
			glUniform1ui(U_RENDER_TYPE, static_cast<GLuint>(traceType % 2));
			glUniform1f(U_SPAN, cache.span);

			// Joint Transforms
			MeshBatchSkeletal* batchPtr = static_cast<MeshBatchSkeletal*>(batches.arr[i]);
			batchPtr->getJointTransforms().BindAsShaderStorageBuffer(LU_JOINT_TRANS);
		}
		else
		{
			// Shader Vert
			vertexDebugVoxel.Bind();
			glUniform1ui(U_RENDER_TYPE, static_cast<GLuint>(traceType % 2));
			glUniform1f(U_SPAN, cache.span);
		}

		cache.cache[i].voxelVAO.Bind();
		cache.cache[i].voxelVAO.Draw(cache.cache[i].batchVoxCacheCount, 0);
	}	
}

void ThesisSolution::DebugRenderVoxelPage(const Camera& camera, 
										  VoxelDebugVAO& pageVoxels,
										  const CVoxelGrid& voxGrid,
										  uint32_t offset,
										  uint32_t voxCount)
{
	//DEBUG VOXEL RENDER
	// Frame Viewport
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0,
			   static_cast<GLsizei>(camera.width),
			   static_cast<GLsizei>(camera.height));
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

	glDisable(GL_MULTISAMPLE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDepthFunc(GL_LESS);
	glDepthMask(true);
	glColorMask(true, true, true, true);

	// Debug Voxelize Pages
	// User World Render Vertex Shader
	Shader::Unbind(ShaderType::GEOMETRY);
	vertexDebugWorldVoxel.Bind();
	glUniform1ui(U_RENDER_TYPE, static_cast<GLuint>(traceType % 2));
	fragmentDebugVoxel.Bind();

	// We need grid info buffer as uniform and frame transform buffer
	cameraTransform.Bind();
	cameraTransform.Update(camera.generateTransform());

	VoxelGridInfoGL voxelGridGL = 
	{
		{voxGrid.position.x, voxGrid.position.y, voxGrid.position.z, voxGrid.span},
		{voxGrid.dimension.x, voxGrid.dimension.y, voxGrid.dimension.z, voxGrid.depth},
	};
	gridInfoBuffer.CPUData()[0] = voxelGridGL;
	gridInfoBuffer.SendData();
	gridInfoBuffer.BindAsUniformBuffer(U_VOXEL_GRID_INFO);

	pageVoxels.Bind();
	pageVoxels.Draw(voxCount, offset);
}

double ThesisSolution::DebugRenderSVO(const Camera& camera)
{
	// Raytrace voxel scene
	SVOTraceType traceTypeEnum = static_cast<SVOTraceType>(traceType % 3);
	double time;
	time = voxelOctree.DebugTraceSVO(dRenderer,
									 camera,
									 svoRenderLevel,
									 traceTypeEnum);
	return time;
}

void ThesisSolution::Frame(const Camera& mainRenderCamera)
{
	// Zero out debug transfer time since it may not be used
	debugVoxTransferTime = 0;

	// VoxelSceneUpdate
	double ioTimeSegment, transformTimeSegment;
	ioTime = 0;
	transformTime = 0;
	svoTime = 0;

	for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
	{
		voxelScenes[i].MapGLPointers();

		// Cascade Update
		voxelScenes[i].VoxelUpdate(ioTimeSegment,
								   transformTimeSegment,
								   mainRenderCamera.pos,
								   static_cast<float>(0x1 << (3 - i - 1)));
		ioTime += ioTimeSegment;
		transformTime += transformTimeSegment;
	}
	ioTime += ioTimeSegment;
	transformTime += transformTimeSegment;
	
	// Octree Update
	svoTime = voxelOctree.UpdateSVO();

	for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
	{
		voxelScenes[i].UnmapGLPointers();
	}

	// Voxel Count in Pages
	for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
	{
		voxelCaches[i].voxOctreeCount = voxelScenes[i].VoxelCountInPage();
		voxelCaches[i].voxOctreeSize = static_cast<double>(voxelCaches[i].voxOctreeCount * sizeof(uint32_t) * 4) / 1024 / 1024;
	}


	aoBar.HideBar(true);
	// Here check TW Bar if user wants to render voxels
	switch(renderScheme)
	{
		case GI_DEFERRED:
		{
			dRenderer.Render(*currentScene, mainRenderCamera, !directLighting);
			break;
		}
		case GI_LIGHT_INTENSITY:
		{
			aoBar.HideBar(false);
				
	//		dRenderer.Render(*currentScene, mainRenderCamera);
	//		dRenderer.ShowLIBuffer(mainRenderCamera);
		
			dRenderer.PopulateGBuffer(*currentScene, mainRenderCamera);
			//debugVoxTransferTime = voxelOctree.AmbientOcclusion
			//(
			//	dRenderer,
			//	mainRenderCamera,
			//	IEMath::ToRadians(aoBar.angleDegree),
			//	aoBar.maxDistance,
			//	aoBar.sampleFactor,
			//	aoBar.intensity
			//);

			debugVoxTransferTime = voxelOctree.GlobalIllumination
			(
				dRenderer,
				mainRenderCamera,
				*currentScene,
				IEMath::ToRadians(aoBar.angleDegree),
				aoBar.maxDistance,
				aoBar.sampleFactor,
				aoBar.intensity
			);

			break;
		}		

		case GI_SVO_DEFERRED:
		{
			SVOTraceType traceTypeEnum = static_cast<SVOTraceType>(traceType % 3);
			dRenderer.PopulateGBuffer(*currentScene, mainRenderCamera);
			debugVoxTransferTime = voxelOctree.DebugDeferredSVO(dRenderer,
																mainRenderCamera,
																svoRenderLevel,
																traceTypeEnum);
			break;
		}

		case GI_SVO_LEVELS:
		{
			// Start Render
			glClearColor(1.0f, 1.0f, 0.0f, 0.0f);
			debugVoxTransferTime = DebugRenderSVO(mainRenderCamera);
			break;
		}
		case GI_VOXEL_PAGE:
		{
			unsigned int totalVoxCount = 0;
			for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
				totalVoxCount += voxelCaches[i].voxOctreeCount;

			voxelNormPosBuffer.Resize(totalVoxCount);
			voxelColorBuffer.Resize(totalVoxCount);

			// Cuda Register	
			CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&vaoNormPosResource, 
												    voxelNormPosBuffer.getGLBuffer(), 
													cudaGraphicsMapFlagsWriteDiscard));
			CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&vaoRenderResource, 
													voxelColorBuffer.getGLBuffer(), 
													cudaGraphicsMapFlagsWriteDiscard));

			CVoxelNormPos* dVoxNormPos = nullptr;
			uchar4* dVoxColor = nullptr;
			size_t bufferSize;
			
			CUDA_CHECK(cudaGraphicsMapResources(1, &vaoNormPosResource));
			CUDA_CHECK(cudaGraphicsMapResources(1, &vaoRenderResource));
			CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dVoxNormPos), 
															&bufferSize,
															vaoNormPosResource));
			CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dVoxColor),
															&bufferSize,
															vaoRenderResource));

			std::vector<uint32_t> offsets;
			std::vector<uint32_t> counts;

			// Create VAO after resize since buffer id can change
			VoxelDebugVAO vao(voxelNormPosBuffer, voxelColorBuffer);

			// Start Render
			glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
			glClear(GL_COLOR_BUFFER_BIT | 
					GL_DEPTH_BUFFER_BIT);
			
			
			uint32_t voxelCount = 0, voxelOffset = 0;
			std::vector<CVoxelGrid> voxGrids(GI_CASCADE_COUNT);
			
			debugVoxTransferTime = 0;
			for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
			{
				offsets.push_back(voxelOffset);
				debugVoxTransferTime += voxelScenes[i].VoxDataToGL(dVoxNormPos + voxelOffset,
																   dVoxColor + voxelOffset,
																   voxGrids[i],
																   voxelCount,
																   voxelCaches[i].voxOctreeCount);
				voxelOffset += voxelCount;
				counts.push_back(voxelCount);
			}
			// All written unmap
			CUDA_CHECK(cudaGraphicsUnmapResources(1, &vaoNormPosResource));
			CUDA_CHECK(cudaGraphicsUnmapResources(1, &vaoRenderResource));
			CUDA_CHECK(cudaGraphicsUnregisterResource(vaoNormPosResource));
			CUDA_CHECK(cudaGraphicsUnregisterResource(vaoRenderResource));

			// Render
			for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
			{
				DebugRenderVoxelPage(mainRenderCamera, 
									 vao, 
									 voxGrids[i], offsets[i], counts[i]);
			}
			break;
		}
		case GI_VOXEL_CACHE2048:
		{
			glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
			DebugRenderVoxelCache(mainRenderCamera, voxelCaches[0]);
			break;
		}
		case GI_VOXEL_CACHE1024:
		{
			glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
			DebugRenderVoxelCache(mainRenderCamera, voxelCaches[1]);
			break;
		}
		case GI_VOXEL_CACHE512:
		{
			glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
			DebugRenderVoxelCache(mainRenderCamera, voxelCaches[2]);
			break;
		}
	}
}

void ThesisSolution::LevelIncrement(void* solPtr)
{
	static_cast<ThesisSolution*>(solPtr)->LevelIncrement();
}

void ThesisSolution::LevelDecrement(void* solPtr)
{
	static_cast<ThesisSolution*>(solPtr)->LevelDecrement();
}

void ThesisSolution::TraceIncrement(void* solutionPtr)
{
	static_cast<ThesisSolution*>(solutionPtr)->TraceTypeInc();
}

void ThesisSolution::TraceDecrement(void* solutionPtr)
{
	static_cast<ThesisSolution*>(solutionPtr)->TraceTypeDec();
}