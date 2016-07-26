
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
OGLTimer::OGLTimer()
	: queryObj(0)
	, timeStamp(0)
{
	glGenQueries(1, &queryObj);
}

OGLTimer::OGLTimer(OGLTimer&& other)
	: queryObj(other.queryObj)
	, timeStamp(other.timeStamp)
{
	other.queryObj = 0;
	timeStamp = 0;
}

OGLTimer& OGLTimer::operator=(OGLTimer&& other)
{
	assert(this != &other);
	queryObj = other.queryObj;
	timeStamp = other.timeStamp;
	other.queryObj = 0;
	other.timeStamp = 0;
	return *this;
}

OGLTimer::~OGLTimer()
{
	glDeleteQueries(1, &queryObj);
}

void OGLTimer::Start()
{
	glBeginQuery(GL_TIME_ELAPSED, queryObj);
}

void OGLTimer::Stop()
{
	glEndQuery(GL_TIME_ELAPSED);
	glGetQueryObjectui64v(queryObj, GL_QUERY_RESULT, &timeStamp);
}

double OGLTimer::ElapsedS() const
{
	return timeStamp / 1000000000.0;
}

double OGLTimer::ElapsedMS() const
{
	return timeStamp / 1000000.0;
}

double OGLTimer::ElapsedUS() const
{
	return timeStamp / 1000.0;
}
#endif //__OGLTIMER_H__