#include <Arduino.h>
#include <Encoder.h>
#include <RotaryEncoder.h>
#include "config.h"

long oldPosition1  = -999;

RotaryEncoder CartEncoder(CartEncoderA,CartEncoderB,RotaryEncoder::LatchMode::TWO03);
RotaryEncoder MotorEncoder(MotorEncoderA,MotorEncoderB,RotaryEncoder::LatchMode::TWO03);

//TODO

void motorForward(int speed){

}

void motorBackward(int speed){

}
void motorStop(){
  
}

void setup() {
  Serial.begin(9600);
  pinMode(MotorPin1,OUTPUT);
  pinMode(MotorPin2,OUTPUT);
}

void loop() {
  if(Serial.available()){
    switch (Serial.read())
    {
    case 'j'://turn forward
      digitalWrite(motor1, HIGH);
      digitalWrite(motor2, LOW);
      break;
    case 'k'://turn backward
      digitalWrite(motor1, LOW);
      digitalWrite(motor2, HIGH);
      break;
    case 'l'://Stop
      digitalWrite(motor1, LOW);
      digitalWrite(motor2, LOW);
      break;
    default:
      break;
    }
  }

  // long newPosition = rotEnc1.read();
  // if (newPosition != oldPosition1) {
  //   oldPosition1 = newPosition;
  //   Serial.println(newPosition);
  // }

 static int pos = 0;
  CartEncoder.tick();

  int newPos = CartEncoder.getPosition();
  if (pos != newPos) {
    Serial.println(newPos);
    pos = newPos;
  }

}