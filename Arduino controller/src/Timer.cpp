#include "Timer.h"
#include "Arduino.h"
Timer::Timer() {
    
}

bool Timer::CheckTimer(int delay){
    currentTime = millis();
    if(currentTime - timer >= delay){
    timer = millis();
    return true;
    }
return false;
}
bool Timer::Delay(int delay){
    currentTime = millis();
    if(timer == 0){
        timer = currentTime;
    }
    if(currentTime - timer >= delay ){
        timer = 0;
        return true;
    }
    return false;
}

void Timer::ResetDelay(){
    timer = 0;
}

void Timer::ResetTimer(){
    timer = millis();
}