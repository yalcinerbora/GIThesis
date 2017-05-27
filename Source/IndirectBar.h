#pragma once

#include "AntBar.h"
#include "IEUtility/IEVector3.h"
#include "SceneLights.h"
#include "RenderSelect.h"

class IndirectBar : public AntBar
{
	private:
		static constexpr char*		IndirectBarName = "IndirectParams";

		float		angleDegree;
		float		sampleFactor;
		float		maxDistance;
		float		falloffFactor;
		float		intensityAO;
		float		intensityGI;
		bool		hidden;
		bool		specular;

	public:
					// Constructors & Destructor
					IndirectBar() = default;
					IndirectBar(SceneLights& sceneLights,
								
								float& diffuseAngle,
								float& specularAngle,

								float& sampleFactor,
								float& maxDistance,
								float& falloffFactor,
								float& intensityAO,
								float& intensityGI,
								bool& hidden,
								bool& specular);
		IndirectBar&			operator=(IndirectBar&&) = default;
								~IndirectBar() = default;
};
