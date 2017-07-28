#pragma once

#include "AntBar.h"
#include "IEUtility/IEVector3.h"
#include "SceneLights.h"
#include "RenderSelect.h"
#include "GISparseVoxelOctree.h"

class IndirectBar : public AntBar
{
	private:
		static constexpr char*		IndirectBarName = "IndirectParams";

	public:
									// Constructors & Destructor
									IndirectBar() = default;
									IndirectBar(SceneLights& sceneLights,
												IndirectUniforms& iUniforms,
												bool& specularOn);
		IndirectBar&				operator=(IndirectBar&&) = default;
									~IndirectBar() = default;
};
