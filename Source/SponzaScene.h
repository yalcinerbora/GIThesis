#pragma once

#include "Scene.h"

class SponzaScene : public ConstantScene
{

	private:
		void								PatrolNyra(double elapsedS);

		static constexpr IEVector3			velocityBase = IEVector3(0.0f, 0.0f, 25.0f);
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