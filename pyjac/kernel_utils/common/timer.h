#ifndef TIMER_H
#define TIMER_H

#include <stdlib.h>

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
 #ifndef __USE_BSD
 #define __USE_BSD
 #endif
#include <time.h>
#include <sys/time.h>
#endif

void StartTimer();
// time elapsed in ms
double GetTimer();

#endif // TIMER_H