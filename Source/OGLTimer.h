
#ifndef __OGLTIMER_H__
#define __OGLTIMER_H__

#include <cassert>
#include "GLHeader.h"

class OGLTimer
{
	private:
		GLuint		queryObj;
		GLuint64	timeStamp;

	protected:
	public:
		// Constructors & Destructor
					OGLTimer();
					OGLTimer(const OGLTimer&) = delete;
					OGLTimer(OGLTimer&&);
		OGLTimer&	operator=(const OGLTimer&) = delete;
		OGLTimer&	operator=(OGLTimer&&);
					~OGLTimer();

		// Usage
		void		Start();
		void		Stop();

		double		ElapsedS() const;
		double		ElapsedMS() const;
		double		ElapsedUS() const;
};

// 
inline OGLTimer::OGLTimer()
	: queryObj(0)
	, timeStamp(0)
{
	glGenQueries(1, &queryObj);
}

inline OGLTimer::OGLTimer(OGLTimer&& other)
	: queryObj(other.queryObj)
	, timeStamp(other.timeStamp)
{
	other.queryObj = 0;
	timeStamp = 0;
}

inline OGLTimer& OGLTimer::operator=(OGLTimer&& other)
{
	glDeleteQueries(1, &queryObj);
	assert(this != &other);
	queryObj = other.queryObj;
	timeStamp = other.timeStamp;
	other.queryObj = 0;
	other.timeStamp = 0;
	return *this;
}

inline OGLTimer::~OGLTimer()
{
	glDeleteQueries(1, &queryObj);
}

inline void OGLTimer::Start()
{
	glBeginQuery(GL_TIME_ELAPSED, queryObj);
}

inline void OGLTimer::Stop()
{
	glEndQuery(GL_TIME_ELAPSED);
	glGetQueryObjectui64v(queryObj, GL_QUERY_RESULT, &timeStamp);
}

inline double OGLTimer::ElapsedS() const
{
	return timeStamp / 1000000000.0;
}

inline double OGLTimer::ElapsedMS() const
{
	return timeStamp / 1000000.0;
}

inline double OGLTimer::ElapsedUS() const
{
	return timeStamp / 1000.0;
}
#endif //__OGLTIMER_H__