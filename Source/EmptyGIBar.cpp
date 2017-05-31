#include "EmptyGIBar.h"

EmptyGIBar::EmptyGIBar(const SceneLights& lights,					   
					   RenderScheme& scheme,
					   double& frameTime,
					   double& shadowTime,
					   double& dPassTime,
					   double& gPassTime,
					   double& lightTime,
					   double& mergeTime)
	: AntBar(EmptyGIBarName)
	, renderSelect(bar, lights, scheme)
{
	// Timings
	TwAddVarRO(bar, "frameTime", TW_TYPE_DOUBLE, &frameTime,
			   " label='Frame' precision=2 help='Total Frame Time.' ");
	TwAddSeparator(bar, NULL, NULL);
	TwAddVarRO(bar, "shadowTime", TW_TYPE_DOUBLE, &shadowTime,
			   " label='Shadow Map' group='Timings (ms)' precision=2 help='Shadow Map generation timing per frame.' ");
	TwAddVarRO(bar, "dPassTime", TW_TYPE_DOUBLE, &dPassTime,
			   " label='Depth Pre-pass' group='Timings (ms)' precision=2 help='Depth Pre-pass timing per frame.' ");
	TwAddVarRO(bar, "gPassTime", TW_TYPE_DOUBLE, &gPassTime,
			   " label='G-pass' group='Timings (ms)' precision=2 help='G-pass timing per frame.' ");
	TwAddVarRO(bar, "lightTime", TW_TYPE_DOUBLE, &lightTime,
			   " label='Light pass' group='Timings (ms)' precision=2 help='Light Pass timing per frame.' ");
	TwAddVarRO(bar, "mergeTime", TW_TYPE_DOUBLE, &mergeTime,
			   " label='Merge' group='Timings (ms)' precision=2 help='Merge timing per frame.' ");

	TwDefine((std::string(EmptyGIBarName) + " refresh=0.01 ").c_str());
	TwDefine((std::string(EmptyGIBarName) + " size='220 180' ").c_str());
	TwDefine((std::string(EmptyGIBarName) + " valueswidth=fit ").c_str());
	TwDefine((std::string(EmptyGIBarName) + " position='5 278' ").c_str());
}

bool EmptyGIBar::DoTiming() const
{
	int opened;
	TwGetParam(bar, "Timings (ms)",
			   "opened", TW_PARAM_INT32, 1, &opened);
	return opened != 0;
}

int EmptyGIBar::Light() const
{
	return renderSelect.Light();
}

int EmptyGIBar::LightLevel() const
{
	return renderSelect.LightLevel();
}

void EmptyGIBar::Next()
{
	renderSelect.Next();
}

void EmptyGIBar::Previous()
{
	renderSelect.Previous();
}

void EmptyGIBar::Up()
{
	renderSelect.Up();
}

void EmptyGIBar::Down()
{
	renderSelect.Down();
}