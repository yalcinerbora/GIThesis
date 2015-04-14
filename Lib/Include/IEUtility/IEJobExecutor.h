/**

Author(s) :
	Bora Yalciner
*/

#ifndef __IE_JOBEXECUTER_H__
#define __IE_JOBEXECUTER_H__

#include "IEWorker.h"

#define IE_DEFAULT_WORKER_AMOUNT 8

template <IESize W = IE_DEFAULT_WORKER_AMOUNT>
class IEJobExecutor
{
	private:
		IEWorker				workers[W];
		int						ChooseWorker();			// Worker Choosing Logic for the upcoming task

	public:
		// Constructors & Destructor
								IEJobExecutor();
								IEJobExecutor(const IEJobExecutor&) = delete;
		virtual					~IEJobExecutor();

		// Thread Lifetime JobExecuter	
		void					Start();
		void					Stop();

		// Function Def Copied From std::async
		template <class Function, class... Args>
		std::future<typename std::result_of<Function(Args...)>::type>
								AssignJob(Function&&, Args&&...);


};
#include "IEJobExecutor.hpp"
#endif //__IE_JOBEXECUTER_H__