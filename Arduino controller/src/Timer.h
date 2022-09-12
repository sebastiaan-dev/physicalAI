#ifndef Timer_h
#define Timer_h
#include "Arduino.h"
class Timer{
    public:
    Timer();
    void init();
    bool CheckTimer(int delay);
    void ResetTimer();
    bool Delay(int delay);
    void ResetDelay();
    private:
    long timer = 0;
    unsigned long currentTime = 0;
    int _delay;
};
#endif