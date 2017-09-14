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

		static constexpr float		DiffuseLo = 0.0f;
		static constexpr float		DiffuseHi = 60.0f;
		static constexpr float		SpecularMinRangeLo = 2.0f;
		static constexpr float		SpecularMinRangeHi = 10.0f;
		static constexpr float		SpecularMaxRangeLo = 10.0f;
		static constexpr float		SpecularMaxRangeHi = 35.0f;
		static constexpr float		SampleRatioLo = 0.5f;
		static constexpr float		SampleRatioHi = 5.0f;
		static constexpr float		OffsetBiasLo = 0.5f;
		static constexpr float		OffsetBiasHi = 5.5f;
		static constexpr float		TotalDistanceLo = 10.0f;
		static constexpr float		TotalDistanceHi = 500.0f;
		static constexpr float		AOIntensityLo = 0.0f;
		static constexpr float		AOIntensityHi = 20.0f;
		static constexpr float		GIIntensityLo = 0.0f;
		static constexpr float		GIIntensityHi = 20.0f;
		static constexpr float		AOFalloffLo = 0.5f;
		static constexpr float		AOFalloffHi = 2.0f;


		static void TW_CALL			SetDiffuseTangent(const void *value, void *clientData);
		static void TW_CALL			GetDiffuseTangent(void *value, void *clientData);
		static void TW_CALL			SetSpecularTangent(const void *value, void *clientData);
		static void TW_CALL			GetSpecularTangent(void *value, void *clientData);

	public:
									// Constructors & Destructor
									IndirectBar() = default;
									IndirectBar(IndirectUniforms& iUniforms,
												bool& specularOn,
												bool& giOn,
												bool& aoOn);
		IndirectBar&				operator=(IndirectBar&&) = default;
									~IndirectBar() = default;
};
