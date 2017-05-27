#pragma once

#include <AntTweakBar.h>
#include "Globals.h"

class SceneLights;

class RenderSelect
{
	public:
	
	private:
		static TwType					twDeferredRenderType;

		int								totalLight;
		std::vector<int>				lightMaxLevel;
		int								currentLight;
		int								currentLightLevel;

	protected:
		static const TwEnumVal			renderSchemeVals[];

		const RenderScheme*				scheme;

		// Protected Constructors
										RenderSelect(TwBar* bar,
													 const SceneLights&,
													 RenderScheme&,
													 TwType);

	public:
		static	void					GenRenderTypeEnum();

		// Constructors & Destructor
										RenderSelect() = default;
										RenderSelect(TwBar* bar,
													 const SceneLights&,
													 RenderScheme&);
										~RenderSelect() = default;

		void							Next();
		void							Previous();
		void							Up();
		void							Down();

		int								CurrentLight() const;
		int								CurrentLightLevel() const;
};

class VoxelRenderSelect : public RenderSelect
{
	private:
		static TwType					twThesisRenderType;

		int								totalCascades;
		int								minSVOLevel;
		int								maxSVOLevel;

		int								currentSVOLevel;
		int								currentCacheCascade;
		int								currentPageLevel;

	protected:
	public:
		static	void					GenRenderTypeEnum();

		// Constructors & Destructor
										VoxelRenderSelect() = default;
										VoxelRenderSelect(TwBar* bar, 
														  const SceneLights&, 
														  RenderScheme&,
														  int totalCascades,
														  int minSVOLevel, int maxSVOLevel);
										~VoxelRenderSelect() = default;

		void							Next();
		void							Previous();
		void							Up();
		void							Down();

		int								CurrentSVOLevel() const;
		int								CurrentCacheCascade() const;
		int								CurrentPageLevel() const;

};
