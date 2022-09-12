#include <Arduino.h>
#include <Encoder.h>
#include <RotaryEncoder.h>  
#include "config.h"
#include "Timer.h"

char * buffer;

Timer cartTimer;
RotaryEncoder CartEncoder(CartEncoderA, CartEncoderB, RotaryEncoder::LatchMode::TWO03);
static int OldCartPos = 0;
static int NewCartPos = 0;

Timer MotorTimer;
RotaryEncoder MotorEncoder(MotorEncoderA, MotorEncoderB, RotaryEncoder::LatchMode::TWO03);
static int OldMotorPos = 0;
static int NewMotorPos = 0;

void motorToLeft(int speed)
{
  //constrain(speed,0,100);
  speed = map(speed,0,100,0,255);
//Serial.print("Speed: ");
//Serial.println(speed);
  analogWrite(MotorPin1, speed);
  analogWrite(MotorPin2, LOW);
}

void motorToRight(int speed)
{
  //constrain(speed,0,100);
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
  motorStop();
 
}

// void loop()
// {
//   //accepting commands
//   if (Serial.available())
//   {
//     memset(buffer,'\0',BufferSize);
//     Serial.readBytesUntil('#',buffer,BufferSize);// F|B|S:0-100#
// //Serial.print("Buffer: ");
// //Serial.println(buffer);
//     int input = 0;
//     switch (*strtok(buffer,":"))
//     {
//     case 'L': // turn forward
//       input = atoi(strtok(NULL,":"));
//       motorToLeft(input);
// Serial.print("Left command recieved: ");
// Serial.println(input);
//       break;
//     case 'R': // turn backward
//       input = atoi(strtok(NULL,":"));
//       motorToRight(input);
// Serial.print("Right command recieved: ");
// Serial.println(input);
//       break;
//     case 'S': // Stop
//       motorStop();
// Serial.println("Stop command recieved");
//       break;
//     default:
// Serial.print("Unknown command: " );
// Serial.println(buffer);
//       break;
//     }
//   }

//   // motorToRight(100);
//   // delay(250);
//   // motorToLeft(100);
//   // delay(250);

//   //relaying encoder positions
//   //motorencoder position

//   // MotorEncoder.tick();

//   // NewMotorPos = MotorEncoder.getPosition();
//   // if (OldMotorPos != NewMotorPos)
//   // {
//   //   Serial.print("M:");
//   //   Serial.println(NewMotorPos);
//   //   OldMotorPos = NewMotorPos;
//   // }

//   // //cartencoder position

//   // CartEncoder.tick();
  
//   // NewCartPos = CartEncoder.getPosition();
//   // if (OldCartPos != NewCartPos)
//   // {
//   //   Serial.print("C:");
//   //   Serial.println(NewCartPos);
//   //   OldCartPos = NewCartPos;
//   // }
// }

void loop(){
  if (Serial.available())
  {
    int input = 100;
    switch (Serial.read())
    {
    case 'L': // turn forward
      motorToLeft(input);
Serial.print("Left command recieved: ");
Serial.println(input);
      break;
    case 'R': // turn backward
      motorToRight(input);
Serial.print("Right command recieved: ");
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
}