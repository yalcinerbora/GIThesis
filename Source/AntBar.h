#pragma once

#include <AntTweakBar.h>

class AntBar
{
	private:
		static const TwStructMember		lightMembers[];

	protected:
		TwBar*							bar;
		const std::string				barName;

	public:		
		// Custom Types
		static TwType					twIEVector3Type;

		// Statics
		static void						InitAntSystem();
		static void						DeleteAntSystem();
		static void						Draw(int windowId);
		static void						ResizeGUI(int windowId, int w, int h);

		static int						KeyCallback(int key, int action);
		static int						MousePosCallback(double x, double y);
		static int						MouseButtonCallback(int button, int action);
		static int						MouseWheelCallback(double offset);
		static void						SetCurrentWindow(int windowId);
		

		// Constructors & Destructor
										AntBar();
										AntBar(const std::string&);
										AntBar(AntBar&&);
										AntBar(const AntBar&) = delete;
		AntBar&							operator=(const AntBar&) = delete;
		AntBar&							operator=(AntBar&&);
										~AntBar();
};