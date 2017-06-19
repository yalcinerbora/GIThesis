#pragma once

#include <AntTweakBar.h>
#include "Globals.h"

class SceneLights;

class RenderSelect
{
	private:
		static TwType					twDeferredRenderType;

		int								totalLight;
		std::vector<int>				lightMaxLevel;
		int								light;
		int								lightLevel;

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

		int								Light() const;
		int								LightLevel() const;
};

class VoxelRenderSelect : public RenderSelect
{
	private:
		static TwType					twThesisRenderType;

		int								totalCascades;
		int								minSVOLevel;
		int								maxSVOLevel;

		int								svoLevel;
		int								cacheCascade;
		int								pageCascade;

		OctreeRenderType				svoRenderType;
		VoxelRenderType					cacheRenderType;
		VoxelRenderType					pageRenderType;

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

		int								SVOLevel() const;
		OctreeRenderType				SVORenderType() const;

		int								CacheCascade() const;
		VoxelRenderType					CacheRenderType() const;

		int								PageCascade() const;
		VoxelRenderType					PageRenderType() const;
		

};
