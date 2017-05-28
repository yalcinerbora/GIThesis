#include "ThesisBar.h"
#include "SceneLights.h"

ThesisBar::ThesisBar(const SceneLights& lights, 
					 RenderScheme& scheme,
					 int totalCascades,
					 int minSVO, int maxSVO)
	: AntBar(ThesisBarName)
	, renderSelect(bar, lights, scheme, totalCascades, minSVO, maxSVO)
	//, totalCascades(totalCascades)
	//, minSVO(minSVO)
	//, maxSVO(maxSVO)
{	
	//// Timings
	//TwAddVarRO(bar, "frameTime", TW_TYPE_DOUBLE, &frameTime,
	//		   " label='Direct' precision=2 help='Total Frame Time.' ");
	//TwAddSeparator(bar, NULL, NULL);
	//TwAddVarRO(bar, "directTime", TW_TYPE_DOUBLE, &directTime,
	//		   " label='Direct' group='Timings (ms)' precision=2 help='Direct Lighting Timing per frame.' ");
	//TwAddVarRO(bar, "ioTime", TW_TYPE_DOUBLE, &ioTime,
	//		   " label='I-O' group='Timings (ms)' precision=2 help='Voxel Include-Exclude Timing per frame.' ");
	//TwAddVarRO(bar, "updateTime", TW_TYPE_DOUBLE, &transTime,
	//		   " label='Transform' group='Timings (ms)' precision=2 help='Voxel Grid Update Timing per frame.' ");
	//TwAddVarRO(bar, "svoReconTime", TW_TYPE_DOUBLE, &svoReconTime,
	//		   " label='SVO Reconstruct' group='Timings (ms)' precision=2 help='SVO Reconstruction Timing per frame.' ");
	//TwAddVarRO(bar, "svoAvgTime", TW_TYPE_DOUBLE, &svoAverageTime,
	//		   " label='SVO Avgerage' group='Timings (ms)' precision=2 help='SVO Average Timing per frame.' ");
	//TwAddVarRO(bar, "coneTraceTime", TW_TYPE_DOUBLE, &coneTraceTime,
	//		   " label='Cone Trace' group='Timings (ms)' precision=2 help='Cone Trace Timing per frame.' ");
	//TwAddVarRO(bar, "miscTime", TW_TYPE_DOUBLE, &coneTraceTime,
	//		   " label='Misc.' group='Timings (ms)' precision=2 help='Misc. Timing per frame.' ");

	//// On off
	//TwAddSeparator(bar, NULL, NULL);	
	//TwAddVarRW(bar, "giOn", TW_TYPE_BOOLCPP,
	//		   &giOn,
	//		   " label='GI On' help='Cone Tracing GI On off' ");
	//TwAddVarRW(bar, "aoOn", TW_TYPE_BOOLCPP,
	//		   &aoOn,
	//		   " label='AO On' help='Cone Tracing AO On off' ");
	//TwAddVarRW(bar, "inject", TW_TYPE_BOOLCPP,
	//		   &injectOn,
	//		   " label='Inject' help='Light Inject On Off' ");

	//// Voxel Counts
	//TwAddSeparator(bar, NULL, NULL);
	////TwAddVarRO(bar, "svoTotalSize", TW_TYPE_STDSTRING, &svoTotalSize,
	////		   "label ='SVO Total' group='Voxel Cache' precision=2 help='Total SVO memory usage in megabytes.'");
	//for(int i = minSVO; i < maxSVO; i++)
	//{
	//	std::string countDef = std::string("label = 'Level") + std::to_string(i) + "' "
	//						   "group='Voxel SVO' help='SVO level voxel count.'";
	//	TwAddVarRO(bar, (std::string("svoLevel") + std::to_string(i)).c_str(), TW_TYPE_UINT32, &svoVoxelCounts[i],
	//			   countDef.c_str());
	//}
	//TwDefine((std::string(ThesisBarName) + "/'Voxel SVO' opened=false ").c_str());
	//TwAddSeparator(bar, NULL, NULL);
	//for(int i = 0; i < totalCascades; i++)
	//{
	//	std::string countDef = std::string("label = 'Cascade#") + std::to_string(i) + "' "
	//						   "group='Voxel Cache' help='Cache voxel count.'";
	//	std::string sizeDef("label ='' group='Voxel Cache' precision=2 help='Cache voxel memory usage in megabytes.'");
	//	TwAddVarRO(bar, (std::string("voxCache") + std::to_string(i)).c_str(), TW_TYPE_UINT32, &cacheCascadeCounts[i],
	//			   countDef.c_str());
	//	TwAddVarRO(bar, (std::string("voxCacheSize") + std::to_string(i)).c_str(), TW_TYPE_STDSTRING, &cacheCascadeSize[i],
	//			   sizeDef.c_str());
	//}
	//TwDefine((std::string(ThesisBarName) + "/'Voxel Cache' opened=false ").c_str());
	//TwAddSeparator(bar, NULL, NULL);
	//for(int i = 0; i < totalCascades; i++)
	//{
	//	std::string countDef = std::string("label = 'Cascade#") + std::to_string(i) + "' "
	//						   "group='Voxel Page' help='Page voxel count.'";
	//	std::string sizeDef("label ='' group='Voxel Page' precision=2 help='Page voxel memory usage in megabytes.'");
	//	TwAddVarRO(bar, (std::string("voxUsed") + std::to_string(i)).c_str(), TW_TYPE_UINT32, &pageCascadeCounts[i],
	//			   countDef.c_str());
	//	TwAddVarRO(bar, (std::string("voxUsedSize") + std::to_string(i)).c_str(), TW_TYPE_STDSTRING, &pageCascadeSize[i],
	//			   sizeDef.c_str());
	//}
	//TwDefine((std::string(ThesisBarName) + " refresh=0.01 ").c_str());
	//TwDefine((std::string(ThesisBarName) + " size='250 400' ").c_str());
	//TwDefine((std::string(ThesisBarName) + " valueswidth=fit ").c_str());
	//TwDefine((std::string(ThesisBarName) + " position='5 278' ").c_str());
}