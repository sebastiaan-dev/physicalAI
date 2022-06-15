#include <Arduino.h>
#include <Encoder.h>
#include <RotaryEncoder.h>  
#include "config.h"

char * buffer;

RotaryEncoder CartEncoder(CartEncoderA, CartEncoderB, RotaryEncoder::LatchMode::TWO03);
static int OldCartPos = 0;
static int NewCartPos = 0;

RotaryEncoder MotorEncoder(MotorEncoderA, MotorEncoderB, RotaryEncoder::LatchMode::TWO03);
static int OldMotorPos = 0;
static int NewMotorPos = 0;

void motorForward(int speed)
{
  //constrain(speed,0,100);
  speed = map(speed,0,100,0,255);
//Serial.print("Speed: ");
//Serial.println(speed);
  analogWrite(MotorPin1, speed);
  analogWrite(MotorPin2, LOW);
}

void motorBackward(int speed)
{
  constrain(speed,0,100);
  speed = map(speed,0,100,0,255);
//Serial.print("Speed: ");
//Serial.println(speed);
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
  buffer = (char *) malloc(BufferSize);
  
}

void loop()
{
  //accepting commands
  if (Serial.available())
  {
    memset(buffer,'\0',BufferSize);
    Serial.readBytesUntil('#',buffer,BufferSize);// F|B|S:0-100#
//Serial.print("Buffer: ");
//Serial.println(buffer);
    int input = 0;
    switch (*strtok(buffer,":"))
    {
    case 'F': // turn forward
      input = atoi(strtok(NULL,":"));
      motorForward(input);
Serial.print("Forward command recieved: ");
Serial.println(input);
      break;
    case 'B': // turn backward
      input = atoi(strtok(NULL,":"));
      motorBackward(input);
Serial.print("Back command recieved: ");
Serial.println(input);
      break;
    case 'S': // Stop
      motorStop();
Serial.println("Stop command recieved");
      break;
    default:
Serial.print("Unknown command: " );
Serial.println(buffer);
      break;
    }
  }

  //relaying encoder positions
  //motorencoder position
  NewMotorPos = MotorEncoder.getPosition();
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