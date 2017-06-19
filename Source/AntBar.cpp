#include "AntBar.h"
#include "IEUtility/IEVector3.h"
#include "RenderSelect.h"

const TwStructMember AntBar::lightMembers[] =
{
	{ "X", TW_TYPE_FLOAT, 0, " help='X' step=0.1 " },
	{ "Y", TW_TYPE_FLOAT, 4, " help='Y' step=0.1 " },
	{ "Z", TW_TYPE_FLOAT, 8, " help='Z' step=0.1 " }
};

TwType AntBar::twIEVector3Type = TwType::TW_TYPE_UNDEF;

// Statics
void AntBar::InitAntSystem()
{
	TwInit(TW_OPENGL_CORE, NULL);

	twIEVector3Type = TwDefineStruct("Vector3", lightMembers, 3,
									 sizeof(IEVector3), NULL, NULL);
	RenderSelect::GenRenderTypeEnum();
	VoxelRenderSelect::GenRenderTypeEnum();
}

void AntBar::DeleteAntSystem()
{
	TwDeleteAllBars();
	TwTerminate();
}

int AntBar::KeyCallback(int key, int action)
{
	return TwEventKeyGLFW(key, action);
}

int AntBar::MousePosCallback(double x, double y)
{
	return TwEventMousePosGLFW(static_cast<int>(x), static_cast<int>(y));
}

int AntBar::MouseButtonCallback(int button, int action)
{
	return TwEventMouseButtonGLFW(button, action);
}

int AntBar::MouseWheelCallback(double offset)
{
	return TwEventMouseWheelGLFW(static_cast<int>(offset));
}

void AntBar::SetCurrentWindow(int windowId)
{
	TwSetCurrentWindow(windowId);
}

void AntBar::Draw(int windowId)
{
	TwSetCurrentWindow(windowId);
	TwDraw();
}

void AntBar::ResizeGUI(int windowId, int w, int h)
{
	TwSetCurrentWindow(windowId);
	TwWindowSize(w, h);
}

AntBar::AntBar()
	: barName("")
	, bar(nullptr)
{}

AntBar::AntBar(const std::string& barName)
	: barName(barName)
	, bar(TwNewBar(barName.c_str()))
{}

AntBar::AntBar(AntBar&& other)
	: barName(std::move(other.barName))
	, bar(other.bar)
{
	other.bar = nullptr;
}

AntBar& AntBar::operator=(AntBar&& other)
{
	if(bar) TwDeleteBar(bar);
	assert(this != &other);
	const_cast<std::string&>(barName) = std::move(other.barName);
	if(bar) TwDeleteBar(bar);
	bar = other.bar;	
	other.bar = nullptr;
	return *this;
}

AntBar::~AntBar()
{
	if(bar) TwDeleteBar(bar);
}
