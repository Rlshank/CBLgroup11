#include <Arduino.h>

#define STEP1  14
#define DIR1   12
#define STEP2  27
#define DIR2   26

#define STEPS  400
#define DELAY  1000

void stepMotor(int stepPin, int dirPin, bool fwd, int steps) {
  digitalWrite(dirPin, fwd ? HIGH : LOW);
  for (int i = 0; i < steps; i++) {
    digitalWrite(stepPin, HIGH); delayMicroseconds(DELAY);
    digitalWrite(stepPin, LOW);  delayMicroseconds(DELAY);
  }
}

void stepBoth(bool fwd1, bool fwd2, int steps) {
  digitalWrite(DIR1, fwd1 ? HIGH : LOW);
  digitalWrite(DIR2, fwd2 ? HIGH : LOW);
  for (int i = 0; i < steps; i++) {
    digitalWrite(STEP1, HIGH); digitalWrite(STEP2, HIGH); delayMicroseconds(DELAY);
    digitalWrite(STEP1, LOW);  digitalWrite(STEP2, LOW);  delayMicroseconds(DELAY);
  }
}

void setup() {
  pinMode(STEP1, OUTPUT); pinMode(DIR1, OUTPUT);
  pinMode(STEP2, OUTPUT); pinMode(DIR2, OUTPUT);
}

void loop() {
  stepMotor(STEP1, DIR1, true,  STEPS); delay(300);
  stepMotor(STEP1, DIR1, false, STEPS); delay(300);
  stepMotor(STEP2, DIR2, true,  STEPS); delay(300);
  stepMotor(STEP2, DIR2, false, STEPS); delay(300);
  stepBoth(true,  true,  STEPS);        delay(300);
  stepBoth(false, false, STEPS);        delay(300);
  stepBoth(true,  false, STEPS);        delay(1000);
}