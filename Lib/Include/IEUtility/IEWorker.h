/**

Worker Class
You can Assign Jobs to it and it will do the tasks
and wait if there is no tasks to process in its internal list

// Reference Implementation & License
https://github.com/progschj/ThreadPool/blob/master/ThreadPool.h
https://github.com/progschj/ThreadPool/blob/master/COPYING

Author(s):
	Bora Yalciner
*/

#ifndef __IE_WORKER_H__
#define __IE_WORKER_H__

#include <future>
#include <functional>
#include <queue>

#include "IEMacros.h"
#include "IEThread.h"
#include "IETypes.h"

// TODO: there is no limit on the queue 
// it may be an issue..
class IEWorker
{
	private:
		IEThread							workerThread;
		bool								stopSignal;

		// Queue and Associated Conc Helpers
		std::queue<std::function<void()>>	assignedJobs;
		mutable IEMutex						mutex;
		mutable IECondVar					conditionVar;

		// Entry Point of the thread
		void								THRDEntryPoint();
		bool								ProcessJob();
		
	protected:
		
	public:
		// Constructor & Destructor
											IEWorker();
											~IEWorker() = default;

		// ThreadLifetime Worker	
		void								Start();
		void								Stop();

		// Return the Amount of Queued Jobs
		int									QueuedJobs() const;

		// Function Def Copied From std::async
		template <class Function, class... Args>
		std::future<typename std::result_of<Function(Args...)>::type>
											AddJob(Function&&, Args&&...);
};

// Template Functions
template <class Function, class... Args>
std::future<typename std::result_of<Function(Args...)>::type>
IEWorker::AddJob(Function&& f, Args&&... args)
{
	typedef typename std::result_of<Function(Args...)>::type returnType;

	// I had to make this by using make_shared
	// I tried to make this without make_shared since it is extra operation but w/e
	auto job = std::make_shared<std::packaged_task<returnType()>>(std::bind(f, args...));
	auto future = job->get_future();
	auto jobWrap = [job]()
	{
		(*job)();
	};

	mutex.lock();
	assignedJobs.push(jobWrap);
	mutex.unlock();
	conditionVar.notify_all();
	return future;
}
#endif //__IE_WORKER_H__