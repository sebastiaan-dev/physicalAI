#include <Arduino.h>
#include <Encoder.h>
#include <RotaryEncoder.h>  
#include "config.h"

RotaryEncoder CartEncoder(CartEncoderA, CartEncoderB, RotaryEncoder::LatchMode::TWO03);
static int OldCartPos = 0;
static int NewCartPos = 0;

RotaryEncoder MotorEncoder(MotorEncoderA, MotorEncoderB, RotaryEncoder::LatchMode::TWO03);
static int OldMotorPos = 0;
static int NewMotorPos = 0;

void motorForward(int speed)
{
  constrain(speed,0,100);
  map(speed,0,100,0,255);
  analogWrite(MotorPin1, speed);
  analogWrite(MotorPin2, LOW);
}

void motorBackward(int speed)
{
  constrain(speed,0,100);
  map(speed,0,100,0,255);
  analogWrite(MotorPin1, LOW);
  analogWrite(MotorPin2, speed);
}
void motorStop()
{
  analogWrite(MotorPin1, LOW);
  analogWrite(MotorPin2, LOW);
}

void calibrateMotorRange(){

}

void setup()
{
  Serial.begin(9600);
  pinMode(MotorPin1, OUTPUT);
  pinMode(MotorPin2, OUTPUT);
}

void loop()
{
  //accepting commands
  if (Serial.available())
  {
    String input = Serial.readString(); // F|B|S:0-100
    
    switch (input.charAt(0))
    {
    case 'F': // turn forward
      motorForward(input.substring(1).toInt());
      break;
    case 'B': // turn backward
      motorBackward(input.substring(1).toInt());
      break;
    case 'S': // Stop
      motorStop();
      break;
    default:
      Serial.println("Unknown command: " + input );
      break;
    }
  }

  //relaying encoder positions
  //motorencoder position
  NewMotorPos = CartEncoder.getPosition();
  if (OldMotorPos != NewMotorPos)
  {
    Serial.println("M:");
    Serial.println(NewMotorPos);
    OldMotorPos = NewMotorPos;
  }
  //cartencoder position
  CartEncoder.tick();

  NewCartPos = CartEncoder.getPosition();
  if (OldCartPos != NewCartPos)
  {
    Serial.println("C:");
    Serial.println(NewCartPos);
    OldCartPos = NewCartPos;
  }
}