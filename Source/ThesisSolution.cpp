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
	, giOn(true)
	, aoOn(true)
	, injectOn(true)
	, specularOn(true)
	, directLighting(true)
	, ambientLighting(true)
	, ambientColor(0.1f, 0.1f, 0.1f)
	, indirectUniforms{SpecularMin,
					   SpecularMax,
					   std::tan(DiffuseAngle * 0.5f),
					   SampleRatio,
					   OffsetBias,
					   TotalDistance,
					   AOIntensity,
					   GIIntensity,
					   AOFalloff}
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
	indirectBar = std::move(IndirectBar(indirectUniforms,
										specularOn, giOn, aoOn));
	lightBar.CollapseLights(true);
	lightBar.Resize(220, 115);
	thesisBar.Move(5, 143);

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
	indirectBar = IndirectBar();
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

	//aoOn = false;
	//directLighting = false;
	//aColor = IEVector3(0.0f);
	//indirectUniforms.startOffset = 20.0f;

	voxelOctree.UpdateSVO(svoReconTime, svoGenPtrTime, svoAverageTime, doTiming,
						  voxelPages, voxelCaches,
						  static_cast<uint32_t>(currentScene->getBatches().size()),
						  liParams,
						  aColor,
						  injectOn);

	// Do GI Pass
	//dRenderer.ClearLI(aColor);
	if(giOn || aoOn)
	{
		// Uniform Updates
		//dRenderer.RefreshFTransform(mainCam);
		voxelOctree.UpdateIndirectUniforms(indirectUniforms);
		voxelOctree.UpdateOctreeUniforms(voxelPages.getOutermostGridPosition());
		dRenderer.RefreshInvFTransform(*currentScene, mainCam,
									   coneTex.Width(), coneTex.Height());

		// GI Cone Trace
		coneTraceTime = 0.0;
		coneTraceTime += voxelOctree.GlobalIllumination(coneTex, dRenderer, 
														mainCam,
														giOn, aoOn, specularOn,
														doTiming);
		// Blur the cone patches
		coneTraceTime += coneTex.BlurTexture(dRenderer.getGBuffer().getDepthGL(), mainCam);
		
		// Application of Indirect Illumination
		dRenderer.ClearLI(IEVector3(0.0f));
		coneTraceTime += voxelOctree.ApplyToLIBuffer(coneTex,
													 dRenderer,
													 giOn, aoOn,
													 doTiming);
	} else dRenderer.ClearLI(aColor);

	// Direct Light Pass
	if(directLighting) dRenderer.LightPass(*currentScene, mainCam, doTiming);

	// Finally Present Buffer
	dRenderer.Present(mainCam, doTiming);
	
	// Direct Lighting Timings
	directTime = dRenderer.ShadowMapTime() + dRenderer.DPassTime() +
		dRenderer.GPassTime() + dRenderer.LPassTime() +
		dRenderer.MergeTime();

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