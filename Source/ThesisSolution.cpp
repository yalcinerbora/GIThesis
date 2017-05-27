#include "ThesisSolution.h"


// Constructors & Destructor
ThesisSolution::ThesisSolution(const std::string& name,
							   WindowInput& inputManager,
							   DeferredRenderer& deferredDenderer)
	: name(name)
	, currentScene(nullptr)
	, dRenderer(deferredDenderer)
	, giOn(false)
	, aoOn(false)
	, injectOn(false)
	, directLighting(true)

{}

bool ThesisSolution::IsCurrentScene(SceneI& scene)
{
	return &scene == currentScene;
}

void ThesisSolution::Load(SceneI&)
{

}

void ThesisSolution::Release()
{

}

void ThesisSolution::Frame(const Camera&)
{

}

void ThesisSolution::SetFPS(double fpsMS)
{

}

void ThesisSolution::Next()
{

}

void ThesisSolution::Previous()
{

}

void ThesisSolution::Up()
{

}

void ThesisSolution::Down()
{

}











//#include "ThesisSolution.h"
//#include "Globals.h"
//#include "SceneI.h"
//#include "DrawBuffer.h"
//#include "Macros.h"
//#include "Camera.h"
//#include "DeferredRenderer.h"
//#include "SceneLights.h"
//#include "IEUtility/IEMath.h"
//#include <cuda_gl_interop.h>
//#include "OGLTimer.h"
//#include "IEUtility/IETimer.h"
//#include "GFGLoader.h"
//#include "MeshBatchSkeletal.h"
//#include <sstream>
//
//const size_t ThesisSolution::InitialObjectGridSize = 512;
//const float ThesisSolution::CascadeSpan = 0.6f;
//const uint32_t ThesisSolution::CascadeDim = 512;
//
////
////AOBar::AOBar()
//// : angleDegree(30.0f)
//// , sampleFactor(1.00f)
//// , maxDistance(300.0f)
//// , falloffFactor(0.25f)
//// , intensityAO(1.40f)
//// , intensityGI(3.60f)
//// , bar(nullptr)
//// , hidden(true)
//// , specular(true)
////{
////	bar = TwNewBar("ConeBar");
////	TwDefine(" ConeBar visible = false ");
////	TwAddVarRW(bar, "cAngle", TW_TYPE_FLOAT, &angleDegree,
////			   " label='Cone Angle' help='Cone Angle' "
////			   " min=1.0 max=90.0 step= 0.01 ");
////	TwAddVarRW(bar, "sFactor", TW_TYPE_FLOAT, &sampleFactor,
////			   " label='Sample Factor' help='Adjusts Sampling Rate' "
////			   " min=0.5 max=10.0 step=0.01 ");
////	TwAddVarRW(bar, "fFactor", TW_TYPE_FLOAT, &falloffFactor,
////			   " label='Falloff Factor' help='Falloff Factor or the cone sample' "
////			   " min=0.01 max=10.0 step=0.01 ");
////	TwAddVarRW(bar, "maxDist", TW_TYPE_FLOAT, &maxDistance,
////			   std::string(" label='Max Distance' help='Maximum Cone Trace Distance' "
////			   " min=" + std::to_string(ThesisSolution::CascadeSpan) + " max=500.0 step=0.1 ").c_str());
////	TwAddVarRW(bar, "intensityAO", TW_TYPE_FLOAT, &intensityAO,
////			   " label='AO Intensity' help='Occlusion Intensity' "
////			   " min=0.5 max=5.0 step=0.01 ");
////	TwAddVarRW(bar, "intensityGI", TW_TYPE_FLOAT, &intensityGI,
////			   " label='GI Intensity' help='Illumination Intensity' "
////			   " min=0.5 max=10.0 step=0.1 ");
////	TwAddVarRW(bar, "specular", TW_TYPE_BOOLCPP, &specular,
////			   " label='Specular' help='Launch Specular Cone' ");
////	TwDefine(" ConeBar size='220 155' ");
////	TwDefine(" ConeBar position='227 25' ");
////	TwDefine(" ConeBar valueswidth=fit ");
////	
////}
////
////void AOBar::HideBar(bool hide)
////{
////	if(hide != hidden)
////	{
////		if(hide)
////			TwDefine(" ConeBar visible=false ");
////		else
////			TwDefine(" ConeBar visible=true ");
////		hidden = hide;
////	}
////}
////
////AOBar::~AOBar()
////{
////	if(bar) TwDeleteBar(bar);
////	bar = nullptr;
////}
//
//ThesisSolution::ThesisSolution(const std::string& name,
//							   DeferredRenderer& dRenderer, const IEVector3& intialCamPos)
//	: EmptyGISolution(name, dRenderer)
//	, vertexDebugVoxel(ShaderType::VERTEX, "Shaders/VoxRender.vert")
//	, vertexDebugVoxelSkeletal(ShaderType::VERTEX, "Shaders/VoxRenderSkeletal.vert")
//	, vertexDebugWorldVoxel(ShaderType::VERTEX, "Shaders/VoxRenderWorld.vert")
//	, fragmentDebugVoxel(ShaderType::FRAGMENT, "Shaders/VoxRender.frag")
//	, bar(nullptr)
//	, renderScheme(GI_VOXEL_PAGE)
//	//, renderScheme(GI_DEFERRED)
//	//, renderScheme(GI_VOXEL_CACHE)
//	, gridInfoBuffer(1)
//	, voxelNormPosBuffer(512)
//	, voxelColorBuffer(512)
//	, voxelOctree()
//	, traceType(0)
//	, aoOn(true)
//	, giOn(true)
//    , injectOn(true)	
//{
//	//renderType = TwDefineEnum("RenderType", renderSchemeVals, GI_END);
//	gridInfoBuffer.AddData({});
//	//for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
//	//{
//	//	voxelScenes.emplace_back(intialCamPos, CascadeSpan * (0x1 << i), CascadeDim);
//	//}
//}
//
//ThesisSolution::~ThesisSolution()
//{}
//
//bool ThesisSolution::IsCurrentScene(SceneI& scene)
//{
//	return &scene == currentScene;
//}
//
//void ThesisSolution::Load(SceneI& s)
//{
//	// Reset GICudaScene
//	voxelCaches.clear();
//	//for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
//	//{
//	//	voxelScenes[i].Reset();
//	//	voxelCaches.emplace_back();
//	//	voxelCaches.back().span = CascadeSpan * (1 << i);
//	//	voxelCaches.back().depth = (GI_CASCADE_COUNT - i - 1) + static_cast<uint32_t>(std::log2(static_cast<float>(CascadeDim)));	
//	//	voxelCaches[i].voxOctreeCount = 0;
//	//	voxelCaches[i].voxOctreeSize = 0;
//	//}
//	EmptyGISolution::Load(s);
//
//	// Voxelization
//	// and Voxel Cache Creation
//	double voxelTotaltime = 0.0;
//	std::vector<MeshBatchI*> batches = currentScene->getBatches();
//	for(unsigned int i = 0; i < batches.size(); i++)
//	{
//		voxelTotaltime += LoadBatchVoxels(batches[i]);
//	}
//
//	for(unsigned int i = 0; i < voxelCaches.size(); i++)
//	{
//		uint32_t totalCount = 0;
//		double totalSize = 0.0f;
//		for(unsigned int j = 0; j < voxelCaches[i].cache.size(); j++)
//		{
//			totalCount += voxelCaches[i].cache[j].batchVoxCacheCount;
//			totalSize += voxelCaches[i].cache[j].batchVoxCacheSize;
//		}
//		voxelCaches[i].totalCacheCount = totalCount;
//		voxelCaches[i].totalCacheSize = totalSize;
//	}
//	GI_LOG("Scene voxelization completed. Elapsed time %f ms", voxelTotaltime);
//	
//	// Voxel Page System Linking
//	//for(unsigned int j = 0; j < GI_CASCADE_COUNT; j++)
//	//	LinkCacheWithVoxScene(voxelScenes[j], voxelCaches[j], 1.0f);
//	
//	// Allocators Link
//	// Ordering is reversed svo tree needs cascades from other to inner
//	//std::vector<GICudaAllocator*> allocators;
//	//for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
//	//	allocators.push_back(voxelScenes[GI_CASCADE_COUNT - i - 1].Allocator());
//
//	//voxelOctree.LinkAllocators(Array32<GICudaAllocator*>{allocators.data(), GI_CASCADE_COUNT},
//	//						   currentScene->SVOLevelSizes());
//	//voxelOctree.LinkSceneShadowMaps(currentScene);
//	//svoRenderLevel = voxelOctree.SVOConsts().totalDepth;
//
//	//// Memory Usage Total
//	//for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
//	//{
//	//	GI_LOG("Voxel Sytem #%d Total Memory Usage %f MB", i,
//	//		   static_cast<double>(voxelScenes[i].AllocatorMemoryUsage()) / 1024.0 / 1024.0);
//	//}
//	//GI_LOG("Voxel Octree Sytem Total Memory Usage %f MB",
//	//	   static_cast<double>(voxelOctree.MemoryUsage()) / 1024.0 / 1024.0);
//
//
//}
//
//void ThesisSolution::Release()
//{
//	EmptyGISolution::Release();
//	//if(bar) TwDeleteBar(bar);
//	//bar = nullptr;
//}
//
//double ThesisSolution::LoadBatchVoxels(MeshBatchI* batch)
//{
//	//IETimer t;
//	//t.Start();
//
//	//// Voxelization
//	//std::stringstream voxPrefix;
//	//voxPrefix << "vox_" << CascadeSpan << "_" << GI_CASCADE_COUNT << "_";
//	//
//	//// Load GFG
//	//std::string batchVoxFile = voxPrefix.str() + batch->BatchName() + ".gfg";
//	//LoadVoxel(voxelCaches, batchVoxFile.c_str(), GI_CASCADE_COUNT,
//	//		  batch->MeshType() == MeshBatchType::RIGID,
// //             batch->RepeatCount());
//
//	//t.Stop();
//	//// Voxel Load Complete
//	//GI_LOG("Loading \"%s\" complete", batchVoxFile.c_str());
//	//GI_LOG("\tDuration : %f ms", t.ElapsedMilliS());
//	//GI_LOG("------");
//	//return t.ElapsedMilliS();
//	return 0.0;
//}
//
//bool ThesisSolution::LoadVoxel(std::vector<SceneVoxCache>& scenes,
//							   const char* gfgFileName, uint32_t cascadeCount,
//							   bool isSkeletal,
//                               int repeatCount)
//{
//	std::ifstream stream(gfgFileName, std::ios_base::in | std::ios_base::binary);
//	GFGFileReaderSTL stlFileReader(stream);
//	GFGFileLoader gfgFile(&stlFileReader);
//
//	GFGFileError e = gfgFile.ValidateAndOpen();
//	assert(e == GFGFileError::OK);
//
//	// Assertions
//	const auto& header = gfgFile.Header();
//	assert((header.meshes.size() - 1) == cascadeCount);
//
//	// First mesh contains objInfos
//	const auto& meshObjCount = header.meshes.back();
//	assert(meshObjCount.components.size() == cascadeCount);
//
//	uint32_t objCount = static_cast<uint32_t>(meshObjCount.headerCore.vertexCount);
//	std::vector<uint8_t> objectInfoData(gfgFile.MeshVertexDataSize(cascadeCount));
//	gfgFile.MeshVertexData(objectInfoData.data(), cascadeCount);
//
//	// Determine VoxelCount
//	for(uint32_t i = 0; i < cascadeCount; i++)
//	{
//		const auto& mesh = header.meshes[i];
//
//		// Special case aabbmin show span count
//		assert(scenes[i].span == mesh.headerCore.aabb.min[0]);
//		scenes[i].cache.emplace_back(mesh.headerCore.vertexCount * repeatCount, objCount * repeatCount, isSkeletal);
//
//		// Load to Mem
//		std::vector<uint8_t> meshData(gfgFile.MeshVertexDataSize(i));
//		gfgFile.MeshVertexData(meshData.data(), i);
//
//		auto& currentCache = scenes[i].cache.back();
//
//		// Object gridInfo
//		const auto& component = meshObjCount.components[i];
//		assert(component.dataType == GFGDataType::UINT32_2);
//		assert(sizeof(ObjGridInfo) == GFGDataTypeByteSize[static_cast<int>(GFGDataType::UINT32_2)]);
//		assert(component.internalOffset == 0);
//		assert(component.logic == GFGVertexComponentLogic::POSITION);
//		assert(component.stride == sizeof(ObjGridInfo));
//
//		currentCache.objInfo.CPUData().resize(objCount * repeatCount);
//        for(int j = 0; j < repeatCount; j++)
//        {
//            std::memcpy(currentCache.objInfo.CPUData().data() + objCount * j,
//                        objectInfoData.data() + component.startOffset,
//                        objCount * component.stride);
//        }
//
//		// Voxel Data
//		for(const auto& component : mesh.components)
//		{
//			if(component.logic == GFGVertexComponentLogic::POSITION)
//			{
//				// NormPos
//				assert(component.dataType == GFGDataType::UINT32_2);
//				auto& normPosVector = currentCache.voxelNormPos.CPUData();
//
//				normPosVector.resize(mesh.headerCore.vertexCount * repeatCount);
//                for(int j = 0; j < repeatCount; j++)
//                {
//                    std::memcpy(normPosVector.data() + mesh.headerCore.vertexCount * j,
//                                meshData.data() + component.startOffset,
//                                mesh.headerCore.vertexCount * component.stride);
//                }
//			}
//			else if(component.logic == GFGVertexComponentLogic::NORMAL)
//			{
//				// Vox Ids
//				assert(component.dataType == GFGDataType::UINT32_2);
//				auto& voxIdsVector = currentCache.voxelIds.CPUData();
//
//				voxIdsVector.resize(mesh.headerCore.vertexCount * repeatCount);
//                for(int j = 0; j < repeatCount; j++)
//                {
//                    std::memcpy(voxIdsVector.data() + mesh.headerCore.vertexCount * j,
//                                meshData.data() + component.startOffset,
//                                mesh.headerCore.vertexCount * component.stride);
//                }
//			}
//			else if(component.logic == GFGVertexComponentLogic::COLOR)
//			{
//				// Color
//				assert(component.dataType == GFGDataType::UNORM8_4);
//				auto& voxColorVector = currentCache.voxelRenderData.CPUData();
//
//				voxColorVector.resize(mesh.headerCore.vertexCount * repeatCount);
//                for(int j = 0; j < repeatCount; j++)
//                {
//                    std::memcpy(voxColorVector.data() + mesh.headerCore.vertexCount * j,
//                                meshData.data() + component.startOffset,
//                                mesh.headerCore.vertexCount * component.stride);
//                }
//			}
//			else if(component.logic == GFGVertexComponentLogic::WEIGHT)
//			{
//				// Weight
//				assert(component.dataType == GFGDataType::UINT32_2);
//				auto& voxWeightVector = currentCache.voxelWeightData.CPUData();
//
//				voxWeightVector.resize(mesh.headerCore.vertexCount * repeatCount);
//                for(int j = 0; j < repeatCount; j++)
//                {
//                    std::memcpy(voxWeightVector.data() + mesh.headerCore.vertexCount * j,
//                                meshData.data() + component.startOffset,
//                                mesh.headerCore.vertexCount * component.stride);
//                }
//			}
//			else
//			{
//				assert(false);
//			}
//			currentCache.voxelRenderData.SendData();
//			currentCache.voxelNormPos.SendData();
//			currentCache.voxelIds.SendData();
//			currentCache.objInfo.SendData();
//			if(isSkeletal) currentCache.voxelWeightData.SendData();
//		}
//
//		GI_LOG("\tCascade#%d Voxels : %zd", i, mesh.headerCore.vertexCount  * repeatCount);
//	}
//	return true;
//}
//
//
//void ThesisSolution::LinkCacheWithVoxScene(GICudaVoxelScene& scene, 
//										   SceneVoxCache& cache,
//										   float coverageRatio)
//{
//	//// Send it to CUDA
//	//std::vector<MeshBatchI*> batches = currentScene->getBatches();
//	//assert(batches.length == cache.cache.size());
//	//for(unsigned int i = 0; i < cache.cache.size(); i++)
//	//{
//	//	GLuint jointBuffer = 0;
//	//	if(batches.arr[i]->MeshType() == VoxelObjectType::SKEL_DYNAMIC)
//	//		jointBuffer = static_cast<MeshBatchSkeletal*>(batches.arr[i])->getJointTransforms().getGLBuffer();
//	//	scene.LinkOGL(batches.arr[i]->getDrawBuffer().getAABBBuffer().getGLBuffer(),
//	//				  batches.arr[i]->getDrawBuffer().getModelTransformBuffer().getGLBuffer(),
//	//				  jointBuffer,
//	//				  batches.arr[i]->getDrawBuffer().getModelTransformIndexBuffer().getGLBuffer(),
//	//				  cache.cache[i].objInfo.getGLBuffer(),
//	//				  cache.cache[i].voxelNormPos.getGLBuffer(),
//	//				  cache.cache[i].voxelIds.getGLBuffer(),
//	//				  cache.cache[i].voxelRenderData.getGLBuffer(),
//	//				  cache.cache[i].voxelWeightData.getGLBuffer(),
//	//				  static_cast<uint32_t>(batches.arr[i]->DrawCount()),
//	//				  cache.cache[i].batchVoxCacheCount);
//	//}
//	//// Allocate at least all of the scene voxel
//	//scene.AllocateWRTLinkedData(coverageRatio);
//}
//
//void ThesisSolution::SVOLevelIncrement()
//{
//	svoRenderLevel++;
//	svoRenderLevel = std::min(svoRenderLevel, voxelOctree.MaxLevel());
//	GI_LOG("Level %d", svoRenderLevel);
//}
//
//void ThesisSolution::SVOLevelDecrement()
//{
//	svoRenderLevel--;
//	svoRenderLevel = std::max(svoRenderLevel, voxelOctree.MinLevel());
//	GI_LOG("Level %d", svoRenderLevel);
//}
//
//void ThesisSolution::TraceTypeIncrement()
//{
//	traceType++;
//	//GI_LOG("Trace Type %d", traceType % 3);
//}
//
//void ThesisSolution::TraceTypeDecrement()
//{
//	traceType--;
//	//GI_LOG("Trace Type %d", traceType % 3);
//}
//
//double ThesisSolution::DebugRenderVoxelCache(const Camera& camera, 
//											 SceneVoxCache& cache)
//{
//	//// Timing
//	//GLuint queryID;
//	//glGenQueries(1, &queryID);
//	//glBeginQuery(GL_TIME_ELAPSED, queryID);
//
//	////DEBUG VOXEL RENDER
//	//// Frame Viewport
//	//glBindFramebuffer(GL_FRAMEBUFFER, 0);
//	//glViewport(0, 0,
//	//		   static_cast<GLsizei>(camera.width),
//	//		   static_cast<GLsizei>(camera.height));
//
//	//glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
//
//	//glDisable(GL_MULTISAMPLE);
//	//glEnable(GL_DEPTH_TEST);
//	//glEnable(GL_CULL_FACE);
//	//glDepthFunc(GL_LEQUAL);
//	//glDepthMask(true);
//	//glColorMask(true, true, true, true);
//
//	//glClear(GL_COLOR_BUFFER_BIT |
//	//		GL_DEPTH_BUFFER_BIT);
//
//	//// Debug Voxelize Scene
//	//Shader::Unbind(ShaderType::GEOMETRY);
//	//fragmentDebugVoxel.Bind();
//
//	//cameraTransform.Bind();
//	//cameraTransform.Update(camera.generateTransform());
//
//	//Array32<MeshBatchI*> batches = currentScene->getBatches();
//	//for(unsigned int i = 0; i < cache.cache.size(); i++)
//	//{		
//	//	DrawBuffer& dBuffer = batches.arr[i]->getDrawBuffer();
//	//	dBuffer.getAABBBuffer().BindAsShaderStorageBuffer(LU_AABB);
//	//	dBuffer.getModelTransformBuffer().BindAsShaderStorageBuffer(LU_MTRANSFORM);
//	//	dBuffer.getModelTransformIndexBuffer().BindAsShaderStorageBuffer(LU_MTRANSFORM_INDEX);
//
//	//	if(batches.arr[i]->MeshType() == VoxelObjectType::SKEL_DYNAMIC)
//	//	{
//	//		// Shader Vert
//	//		vertexDebugVoxelSkeletal.Bind();
//	//		glUniform1ui(U_RENDER_TYPE, static_cast<GLuint>(traceType % 2));
//	//		glUniform1f(U_SPAN, cache.span);
//
//	//		// Joint Transforms
//	//		MeshBatchSkeletal* batchPtr = static_cast<MeshBatchSkeletal*>(batches.arr[i]);
//	//		batchPtr->getJointTransforms().BindAsShaderStorageBuffer(LU_JOINT_TRANS);
//	//	}
//	//	else
//	//	{
//	//		// Shader Vert
//	//		vertexDebugVoxel.Bind();
//	//		glUniform1ui(U_RENDER_TYPE, static_cast<GLuint>(traceType % 2));
//	//		glUniform1f(U_SPAN, cache.span);
//	//	}
//
//	//	cache.cache[i].voxelVAO.Bind();
//	//	cache.cache[i].voxelVAO.Draw(cache.cache[i].batchVoxCacheCount, 0);
//	//}
//
//	//// Timer
//	//GLuint64 timeElapsed = 0;
//	//glEndQuery(GL_TIME_ELAPSED);
//	//glGetQueryObjectui64v(queryID, GL_QUERY_RESULT, &timeElapsed);
//	//return timeElapsed / 1000000.0;
//	return 0.0;
//}
//
//void ThesisSolution::DebugRenderVoxelPage(const Camera& camera, 
//										  VoxelDebugVAO& pageVoxels,
//										  const CVoxelGrid& voxGrid,
//										  uint32_t offset,
//										  uint32_t voxCount)
//{
////	//DEBUG VOXEL RENDER
////	// Frame Viewport
////	glBindFramebuffer(GL_FRAMEBUFFER, 0);
////	glViewport(0, 0,
////			   static_cast<GLsizei>(camera.width),
////			   static_cast<GLsizei>(camera.height));
////	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
////
////	glDisable(GL_MULTISAMPLE);
////	glEnable(GL_DEPTH_TEST);
////	glEnable(GL_CULL_FACE);
////	glDepthFunc(GL_LESS);
////	glDepthMask(true);
////	glColorMask(true, true, true, true);
////
////	// Debug Voxelize Pages
////	// User World Render Vertex Shader
////	Shader::Unbind(ShaderType::GEOMETRY);
////	vertexDebugWorldVoxel.Bind();
////	glUniform1ui(U_RENDER_TYPE, static_cast<GLuint>(traceType % 2));
////	fragmentDebugVoxel.Bind();
////
////	// We need grid info buffer as uniform and frame transform buffer
////	cameraTransform.Bind();
////	cameraTransform.Update(camera.generateTransform());
////
////	VoxelGridInfoGL voxelGridGL = 
////	{
////		{voxGrid.position.x, voxGrid.position.y, voxGrid.position.z, voxGrid.span},
////		{voxGrid.dimension.x, voxGrid.dimension.y, voxGrid.dimension.z, voxGrid.depth},
////	};
////	gridInfoBuffer.CPUData()[0] = voxelGridGL;
////	gridInfoBuffer.SendData();
////	gridInfoBuffer.BindAsUniformBuffer(U_VOXEL_GRID_INFO);
////
////	pageVoxels.Bind();
////	pageVoxels.Draw(voxCount, offset);
//}
//
//double ThesisSolution::DebugRenderSVO(const Camera& camera)
//{
//	//	// Raytrace voxel scene
//	//	SVOTraceType traceTypeEnum = static_cast<SVOTraceType>(traceType % 3);
//	//	double time;
//	//	time = voxelOctree.DebugTraceSVO(dRenderer,
//	//									 camera,
//	//									 svoRenderLevel,
//	//									 traceTypeEnum);
//	//	return time;
//	return 0.0;
//}
//
//void ThesisSolution::Frame(const Camera& mainRenderCamera)
//{
////	// Zero out debug transfer time since it may not be used
////	miscTime = 0.0;
////
////	// VoxelSceneUpdate
////	double ioTimeSegment = 0.0, transformTimeSegment = 0.0;
////	ioTime = 0.0;
////	transformTime = 0.0;
////	svoReconTime = 0.0;
////	svoInjectTime = 0.0;
////	svoAvgTime = 0.0;
////	giTime = 0.0;
////
////	IEVector3 outerCascadePos;
////    for(unsigned int i = 0; i <  GI_CASCADE_COUNT; i++)
////	{
////		voxelScenes[i].MapGLPointers();
////
////        //IEVector3 pos = IEVector3::ZeroVector;
////		// Cascade Update
////		IEVector3 pos = voxelScenes[i].VoxelUpdate(ioTimeSegment,
////												   transformTimeSegment,
////												   mainRenderCamera.pos,
////												   static_cast<float>(0x1 << (3 - i - 1)));
////		ioTime += ioTimeSegment;
////		transformTime += transformTimeSegment;
////		if(i == GI_CASCADE_COUNT - 1) outerCascadePos = pos;
////		//if(i == 0) outerCascadePos = pos;
////	}
////	ioTime += ioTimeSegment;
////	transformTime += transformTimeSegment;
////	
////	// Octree Update
////	// TODO Light Inject
////	IEVector3 camDir = (mainRenderCamera.centerOfInterest - mainRenderCamera.pos).NormalizeSelf();
////	IEVector3 camPos = mainRenderCamera.pos;
////
////	InjectParams p;
////	p.camDir = {camDir.getX(), camDir.getY(), camDir.getZ()};
////	p.camPos = {camPos.getX(), camPos.getY(), camPos.getZ(), DeferredRenderer::CalculateCascadeLength(mainRenderCamera.far, 0)};
////	float depthRange[2];
////	glGetFloatv(GL_DEPTH_RANGE, depthRange);
////	p.depthNear = depthRange[0];
////	p.depthFar = depthRange[1];
////	p.lightCount = currentScene->getSceneLights().Count();
////	p.outerCascadePos = {outerCascadePos.getX(), outerCascadePos.getY(), outerCascadePos.getZ()};
////	p.span = CascadeSpan;
////	p.inject = injectOn;
////	
////	IEVector3 aColor = ambientLighting ? ambientColor : IEVector3::ZeroVector;
////
////	const auto& lightProjs = currentScene->getSceneLights().GetLightProjMatrices();
////	const auto& lightInvVP = currentScene->getSceneLights().GetLightInvViewProjMatrices();
////	voxelOctree.UpdateSVO(svoReconTime, svoInjectTime, svoAvgTime, aColor, 
////						  p, lightProjs, lightInvVP);
////
////	for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
////	{
////		voxelScenes[i].UnmapGLPointers();
////	}
////
////	// Voxel Count in Pages
////	for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
////	{
////		voxelCaches[i].voxOctreeCount = voxelScenes[i].VoxelCountInPage();
////		voxelCaches[i].voxOctreeSize = static_cast<double>(voxelCaches[i].voxOctreeCount * sizeof(uint32_t) * 4) / 1024 / 1024;
////	}
////
////
////	aoBar.HideBar(true);
////	// Here check TW Bar if user wants to render voxels
////	switch(renderScheme)
////	{
////		case GI_DEFERRED:
////		case GI_LIGHT_INTENSITY:
////		{
////			aoBar.HideBar(false);
////			
////			// Shadow Map Generation
////			dRenderer.GenerateShadowMaps(*currentScene, mainRenderCamera);
////
////			// GPass
////			dRenderer.PopulateGBuffer(*currentScene, mainRenderCamera);
////
////			// Clear LI
////			dRenderer.ClearLI(aColor);
////
////			// AO GI Pass
////			if(aoOn || giOn)
////			{
////				giTime = voxelOctree.GlobalIllumination
////				(
////					dRenderer,
////					mainRenderCamera,
////					*currentScene,
////					IEMath::ToRadians(aoBar.angleDegree),
////					aoBar.maxDistance,
////					aoBar.falloffFactor,
////					aoBar.sampleFactor,
////					aoBar.intensityAO,
////					aoBar.intensityGI,
////					giOn,
////					aoOn,
////					aoBar.specular
////				);
////			}
////
////			// Light Pass
////			if(directLighting)
////			{
////				dRenderer.LightPass(*currentScene, mainRenderCamera);
////			}
////
////			// Light Intensity Merge
////			if(renderScheme == GI_DEFERRED)
////			{
////				dRenderer.Present(mainRenderCamera);
////			}
////			else if(renderScheme == GI_LIGHT_INTENSITY)
////			{
////				dRenderer.ShowLIBuffer(mainRenderCamera);
////			}
////			break;
////		}
////		case GI_SVO_DEFERRED:
////		{
////			SVOTraceType traceTypeEnum = static_cast<SVOTraceType>(traceType % 3);
////			dRenderer.PopulateGBuffer(*currentScene, mainRenderCamera);
////			miscTime = voxelOctree.DebugDeferredSVO(dRenderer,
////													mainRenderCamera,
////													svoRenderLevel,
////													traceTypeEnum);
////			break;
////		}
////		case GI_SVO_LEVELS:
////		{
////			// Shadow Map Generation
////			dRenderer.GenerateShadowMaps(*currentScene, mainRenderCamera);
////
////			// Start Render
////			glClearColor(1.0f, 1.0f, 0.0f, 0.0f);
////            glClear(GL_COLOR_BUFFER_BIT);
////			miscTime = DebugRenderSVO(mainRenderCamera);
////			break;
////		}
////		case GI_VOXEL_PAGE:
////		{
////			unsigned int totalVoxCount = 0;
////			for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
////				totalVoxCount += voxelCaches[i].voxOctreeCount;
////
////			if(totalVoxCount == 0) break;
////
////			voxelNormPosBuffer.Resize(totalVoxCount);
////			voxelColorBuffer.Resize(totalVoxCount);
////
////			// Cuda Register	
////			CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&vaoNormPosResource, 
////												    voxelNormPosBuffer.getGLBuffer(), 
////													cudaGraphicsMapFlagsWriteDiscard));
////			CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&vaoRenderResource, 
////													voxelColorBuffer.getGLBuffer(), 
////													cudaGraphicsMapFlagsWriteDiscard));
////
////			CVoxelNormPos* dVoxNormPos = nullptr;
////			uchar4* dVoxColor = nullptr;
////			size_t bufferSize;
////			
////			CUDA_CHECK(cudaGraphicsMapResources(1, &vaoNormPosResource));
////			CUDA_CHECK(cudaGraphicsMapResources(1, &vaoRenderResource));
////			CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dVoxNormPos), 
////															&bufferSize,
////															vaoNormPosResource));
////			CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dVoxColor),
////															&bufferSize,
////															vaoRenderResource));
////
////			std::vector<uint32_t> offsets;
////			std::vector<uint32_t> counts;
////
////			// Create VAO after resize since buffer id can change
////			VoxelDebugVAO vao(voxelNormPosBuffer, voxelColorBuffer);
////
////			// Start Render
////			glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
////			glClear(GL_COLOR_BUFFER_BIT | 
////					GL_DEPTH_BUFFER_BIT);
////			
////			
////			uint32_t voxelCount = 0, voxelOffset = 0;
////			std::vector<CVoxelGrid> voxGrids(GI_CASCADE_COUNT);
////			
////			miscTime = 0;
////			for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
////			{
////				offsets.push_back(voxelOffset);
////				miscTime += voxelScenes[i].VoxDataToGL(dVoxNormPos + voxelOffset,
////													   dVoxColor + voxelOffset,
////													   voxGrids[i],
////													   voxelCount,
////													   voxelCaches[i].voxOctreeCount);
////				voxelOffset += voxelCount;
////				counts.push_back(voxelCount);
////			}
////			// All written unmap
////			CUDA_CHECK(cudaGraphicsUnmapResources(1, &vaoNormPosResource));
////			CUDA_CHECK(cudaGraphicsUnmapResources(1, &vaoRenderResource));
////			CUDA_CHECK(cudaGraphicsUnregisterResource(vaoNormPosResource));
////			CUDA_CHECK(cudaGraphicsUnregisterResource(vaoRenderResource));
////
////			// Render
////			for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
////			{
////				DebugRenderVoxelPage(mainRenderCamera, 
////									 vao, 
////									 voxGrids[i], offsets[i], counts[i]);
////			}
////			break;
////		}
////		case GI_VOXEL_CACHE:
////		{
////			uint32_t level = voxelOctree.MaxLevel() - svoRenderLevel;
////			level = std::min(level, GI_CASCADE_COUNT - 1u);
////
////			glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
////			miscTime = DebugRenderVoxelCache(mainRenderCamera, voxelCaches[level]);
////			break;
////		}
////	}
////	totalTime = ioTime + transformTime + svoReconTime + svoInjectTime +
////				+ svoAvgTime + giTime + miscTime;
//}