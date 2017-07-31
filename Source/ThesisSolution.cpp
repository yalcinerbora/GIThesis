#include "ThesisSolution.h"
#include "DeferredRenderer.h"
#include "SceneI.h"
#include "WindowInput.h"
#include <GLFW\glfw3.h>
#include "Macros.h"

// Constructors & Destructor
ThesisSolution::ThesisSolution(uint32_t denseLevel,
							   uint32_t denseLevelCount,
							   uint32_t cascadeCount,
							   uint32_t cascadeBaseLevel,
							   float baseSpan,
							   WindowInput& inputManager,
							   DeferredRenderer& deferredDenderer,
							   const std::string& name)
	: octreeParams(denseLevel, denseLevelCount,
				   cascadeCount, cascadeBaseLevel,
				   baseSpan)
	, name(name)
	, coneTex(TraceWidth, TraceHeight, GL_RGBA16F)
	, currentScene(nullptr)
	, dRenderer(deferredDenderer)
	, giOn(false)
	, aoOn(false)
	, injectOn(false)
	, directLighting(true)
	, ambientLighting(true)
	, ambientColor(0.1f, 0.1f, 0.1f)
{
	inputManager.AddKeyCallback(GLFW_KEY_KP_ADD, GLFW_RELEASE, &ThesisSolution::Up, this);
	inputManager.AddKeyCallback(GLFW_KEY_KP_SUBTRACT, GLFW_RELEASE, &ThesisSolution::Down, this);
	inputManager.AddKeyCallback(GLFW_KEY_KP_MULTIPLY, GLFW_RELEASE, &ThesisSolution::Next, this);
	inputManager.AddKeyCallback(GLFW_KEY_KP_DIVIDE, GLFW_RELEASE, &ThesisSolution::Previous, this);
}

bool ThesisSolution::IsCurrentScene(SceneI& scene)
{
	return &scene == currentScene;
}

void ThesisSolution::Load(SceneI& s)
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	currentScene = &s;

	// Attach new Scene's Light Indices
	dRenderer.AttachSceneLightIndices(s);

	// Load Voxel Caches
	const auto& batches = currentScene->getBatches();
	std::vector<std::vector<std::string>> batchNames;
	for(int i = 0; i < batches.size(); i++)
	{
		batchNames.push_back(currentScene->getBatchFileNames(i));
	}
	voxelCaches = GIVoxelCache(octreeParams.BaseSpan,
							   octreeParams.CascadeCount,
							   &currentScene->getBatches(),
							   batchNames);

	// Initialize Voxel Page System
	voxelPages = GIVoxelPages(voxelCaches,
							  &currentScene->getBatches(), 
							  octreeParams);

	// Initialize SVO System
	voxelOctree = GISparseVoxelOctree(octreeParams,
									  currentScene,
									  BigSizes);	
	// Initialize GUI
	lightBar = std::move(LightBar(currentScene->getSceneLights(),
								  directLighting,
								  ambientLighting,
								  ambientColor));
	thesisBar = std::move(ThesisBar(currentScene->getSceneLights(),
									scheme,
									frameTime,
									directTime,
									ioTime,
									transTime,
									svoReconTime,
									svoGenPtrTime,
									svoAverageTime,
									coneTraceTime,
									miscTime,
									octreeParams.CascadeCount,
									octreeParams.MinSVOLevel,
									octreeParams.MaxSVOLevel));

	// Indirect Bar
	// TODO:


	// Print System Memory Usage
	GI_LOG("Page Memory Usage %.2fMB", 
		   static_cast<double>(voxelPages.MemoryUsage()) / 1024.0f / 1024.0f);
	GI_LOG("SVO Memory Usage %.2fMB", 
		   static_cast<double>(voxelOctree.MemoryUsage()) / 1024.0f / 1024.0f);
}

void ThesisSolution::Release()
{
	voxelCaches = GIVoxelCache();
	lightBar = LightBar();
	thesisBar = ThesisBar();
	voxelOctree = GISparseVoxelOctree();
	currentScene = nullptr;
}

void ThesisSolution::Frame(const Camera& mainCam)
{
	bool doTiming = thesisBar.DoTiming();
	IEVector3 aColor = ambientLighting ? ambientColor : IEVector3::ZeroVector;

	// Pre-Lighting Operations
	dRenderer.RefreshFTransform(mainCam);
	currentScene->getSceneLights().SendLightDataToGPU();
	dRenderer.GenerateShadowMaps(*currentScene, mainCam, doTiming);
	dRenderer.PopulateGBuffer(*currentScene, mainCam, doTiming);

	// Do Page update
	bool useCache = true;
	voxelPages.Update(ioTime,
					  transTime,
					  voxelCaches,
					  mainCam.pos,
					  doTiming,
					  useCache);

	// Do SVO update
	float depthRange[2];
	glGetFloatv(GL_DEPTH_RANGE, depthRange);
	IEVector3 camDir = (mainCam.centerOfInterest - mainCam.pos).Normalize();
	LightInjectParameters liParams = 
	{
		{mainCam.pos[0], mainCam.pos[1], mainCam.pos[2], currentScene->getSceneLights().getCascadeLength(mainCam.far)},
		{camDir[0], camDir[1], camDir[2]},
		depthRange[0], depthRange[1]
	};
	injectOn = true;
	voxelOctree.UpdateSVO(svoReconTime, svoGenPtrTime, svoAverageTime, doTiming,
						  voxelPages, voxelCaches,
						  static_cast<uint32_t>(currentScene->getBatches().size()),
						  liParams,
						  aColor,
						  injectOn);

	// Do GI
	if(giOn && aoOn)
	{
		coneTraceTime = 0.0;

		//voxelOctree.UpdateIndirectUniforms(indirectUniforms);

		//coneTraceTime += voxelOctree.GlobalIllumination(coneTex, dRenderer, mainCam,
		// aColor,
		//							   giOn, aoOn, specularOn);

		//coneTraceTime += coneTex.BlurTexture(dRenderer.getGBuffer().getDepthGL(), mainCam);

		//// TODO: Incorporate to light buffer
		//dRenderer.ShowTexture(mainCam, coneTex.Texture());
		//// coneTraceTime += dRenderer.AccumulateLIBuffer(coneTex, ...);
	}
	else dRenderer.ClearLI(aColor);
	if(directLighting) dRenderer.LightPass(*currentScene, mainCam, doTiming);

	// Finally Present Buffer
	dRenderer.Present(mainCam, doTiming);
	
	// Direct Lighting Timings
	directTime = dRenderer.ShadowMapTime() + dRenderer.DPassTime() +
		dRenderer.GPassTime() + dRenderer.LPassTime() +
		dRenderer.MergeTime();


	// Rendering Choice
	if(scheme >= RenderScheme::G_DIFF_ALBEDO &&
	   scheme <= RenderScheme::G_DEPTH)
	{
		dRenderer.ShowGBufferTexture(mainCam, scheme);
	}
	else if(scheme == RenderScheme::LIGHT_INTENSITY)
	{
		dRenderer.ShowLightIntensity(mainCam);
	}
	else if(scheme == RenderScheme::SHADOW_MAP)
	{
		dRenderer.ShowShadowMap(mainCam, *currentScene,
								thesisBar.Light(),
								thesisBar.LightLevel());
	}
	else if(scheme == RenderScheme::VOXEL_CACHE)
	{
		voxelCaches.AllocateGL(thesisBar.CacheCascade());
		miscTime = voxelCaches.Draw(doTiming, mainCam, 
									thesisBar.CacheRenderType());
	}
	else
	{
		voxelCaches.DeallocateGL();
		if(scheme == RenderScheme::VOXEL_PAGE)
		{
			voxelPages.AllocateDraw();
			miscTime = voxelPages.Draw(doTiming,
									   thesisBar.PageCascade(),
									   thesisBar.PageRenderType(),
									   mainCam,
									   voxelCaches);
		}
		else 
		{
			voxelPages.DeallocateDraw();
			if(scheme == RenderScheme::SVO_VOXELS)
			{
				// Uniform Updates
				//dRenderer.RefreshFTransform(mainCam);
				dRenderer.RefreshInvFTransform(*currentScene, mainCam,
											   coneTex.Width(), coneTex.Height());
				voxelOctree.UpdateOctreeUniforms(voxelPages.getOutermostGridPosition());

				// Actual Render
				miscTime = voxelOctree.DebugTraceSVO(coneTex, dRenderer, mainCam,
													 thesisBar.SVOLevel(),
													 thesisBar.SVORenderType());

				dRenderer.ShowTexture(mainCam, coneTex.Texture());
			}
			else if(scheme == RenderScheme::SVO_SAMPLE)
			{
				// Uniform Updates
				//dRenderer.RefreshFTransform(mainCam);
				dRenderer.RefreshInvFTransform(*currentScene, mainCam,
											   coneTex.Width(), coneTex.Height());
				voxelOctree.UpdateOctreeUniforms(voxelPages.getOutermostGridPosition());

				// Actual Render
				miscTime = voxelOctree.DebugSampleSVO(coneTex, dRenderer, mainCam,
													  thesisBar.SVOLevel(),
													  thesisBar.SVORenderType());
			
				dRenderer.ShowTexture(mainCam, coneTex.Texture());
			}
		}
	}	
}

void ThesisSolution::SetFPS(double fpsMS)
{
	frameTime = fpsMS;
}

const std::string& ThesisSolution::Name() const
{
	return name;
}

void ThesisSolution::Next()
{
	if(currentScene) thesisBar.Next();
}

void ThesisSolution::Previous()
{
	if(currentScene) thesisBar.Previous();
}

void ThesisSolution::Up()
{
	if(currentScene) thesisBar.Up();
}

void ThesisSolution::Down()
{
	if(currentScene) thesisBar.Down();
}