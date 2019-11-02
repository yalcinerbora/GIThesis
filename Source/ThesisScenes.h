#pragma once

#include "Scene.h"

class CornellScene : public ConstantScene
{

	private:

	protected:

	public:
		// Constructors & Destructor
											CornellScene(const std::string& name,
														 const std::vector<std::string>& rigidFileNames,
														 const std::vector<std::string>& skeletalFileNames,
														 const std::vector<Light>& lights);
											CornellScene(const CornellScene&) = delete;
		CornellScene&						operator=(const CornellScene&) = delete;
											~CornellScene() = default;

		void								Update(double elapsedS) override;
};

class SponzaScene : public ConstantScene
{
	private:
		void								PatrolNyra(double elapsedS);

		static constexpr IEVector3			velocityBase = IEVector3(0.0f, 0.0f, 13.00f);
		static constexpr IEVector3			initalPosBase = IEVector3(0.0f, 0.0f, -4.33f);
		static const IEQuaternion			initalOrientation;

		IEVector3							currentPos;
		IEQuaternion						currentOrientation;
		IEVector3							initalPos;
		IEVector3							velocity;

	protected:

	public:
		// Constructors & Destructor
											SponzaScene(const std::string& name,
														 const std::vector<std::string>& rigidFileNames,
														 const std::vector<std::string>& skeletalFileNames,
														 const std::vector<Light>& lights);
											SponzaScene(const SponzaScene&) = delete;
		SponzaScene&						operator=(const SponzaScene&) = delete;
											~SponzaScene() = default;

		void								Initialize() override;
		void								Update(double elapsedS) override;
};

class DynoScene : public ConstantScene
{
	private:
		static constexpr float				xStart = -150.0f;
		static constexpr float				zStart = -150.0f;
		static constexpr float				distance = 20.0f;
		static constexpr float				width = 300.0f;

		const int							repeatCount;

	protected:
	public:
		// Constructors & Destructor
											DynoScene(const std::string& name,
													  const std::vector<std::string>& rigidFileNames,
													  const std::vector<std::string>& skeletalFileNames,
													  const std::vector<Light>& lights,
													  int repeatCount);
											DynoScene(const DynoScene&) = delete;
		DynoScene&							operator=(const DynoScene&) = delete;
											~DynoScene() = default;

		void								Load() override;
		void								Initialize() override;
		void								Update(double elapsedS) override;
};