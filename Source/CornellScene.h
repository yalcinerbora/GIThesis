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