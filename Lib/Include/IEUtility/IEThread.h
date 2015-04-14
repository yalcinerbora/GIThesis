/**
asdasdasd123

// Barrier Implementation
http://www.boost.org/doc/libs/1_53_0/boost/thread/barrier.hpp

Author(s):
	Bora Yalciner
*/
#ifndef __IE_THREAD_H__
#define __IE_THREAD_H__

#include "IEMacros.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

typedef std::thread IEThread;
typedef std::mutex	IEMutex;
typedef std::recursive_mutex IERecMutex;
typedef std::condition_variable IECondVar;

class IEBarrier
{
	private:	
		unsigned int		count;
		unsigned int		threshold;
		unsigned int		generation;
		IECondVar			condVariable;
		IEMutex				mutex;

	protected:

	public:
		// Constructors & Destructor
							IEBarrier(unsigned int totalCount) : count(totalCount),
																	threshold(totalCount),
																	generation(0)
							{
								if(totalCount <= 1)
									IE_ERROR("Error! Barrier Initialized with less then 2.");
							}

		void				Wait()
							{
								std::unique_lock<std::mutex> lock(mutex);
								unsigned int gen = generation;

								count --;
								if(count == 0)
								{
									generation++;
									count = threshold;
									condVariable.notify_all();
									return;
								}

								while(gen == generation)
									condVariable.wait(lock);
								return;
							}

		void				ForceWakeUp()
							{
								std::unique_lock<std::mutex> lock(mutex);
								generation++;
								count = threshold;
								condVariable.notify_all();
							}
};

// Atomic Vars
typedef std::atomic<bool> IEAtomicBool;
typedef std::atomic<int> IEAtomicInt;

#endif //__IE_THREAD_H__