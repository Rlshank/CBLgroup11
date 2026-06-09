#include <Wire.h>
#include <INA3221.h>

INA3221 ina3221(0x40);

// =======================
// Pin settings
// =======================
#define JOY_X A1
#define JOY_Y A0

#define STEP1 8
#define DIR1 7

#define STEP2 3
#define DIR2 4

// INA channel that worked in your test
#define INA_CHANNEL_1 0

// =======================
// Motor variables
// =======================
int dir1 = 0;
int dir2 = 0;

unsigned long delay1_us = 0;
unsigned long delay2_us = 0;

unsigned long lastStep1 = 0;
unsigned long lastStep2 = 0;

bool stepState1 = LOW;
bool stepState2 = LOW;

// =======================
// Timing for sending sensor data
// =======================
unsigned long lastSend = 0;
const unsigned long SEND_INTERVAL = 50;  // ms

void setup() {
  Serial.begin(115200);

  pinMode(STEP1, OUTPUT);
  pinMode(DIR1, OUTPUT);

  pinMode(STEP2, OUTPUT);
  pinMode(DIR2, OUTPUT);

  digitalWrite(STEP1, LOW);
  digitalWrite(STEP2, LOW);

  Wire.begin();
  ina3221.begin();

  delay(1000);

  Serial.println("READY");
}

void loop() {
  readPythonCommand();
  moveMotors();
  sendDataToPython();
}

void readPythonCommand() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    int values[4];
    int index = 0;

    char buffer[40];
    line.toCharArray(buffer, 40);

    char *token = strtok(buffer, ",");

    while (token != NULL && index < 4) {
      values[index] = atoi(token);
      index++;
      token = strtok(NULL, ",");
    }

    if (index == 4) {
      dir1 = values[0];
      delay1_us = values[1];

      dir2 = values[2];
      delay2_us = values[3];

      digitalWrite(DIR1, dir1);
      digitalWrite(DIR2, dir2);
    }
  }
}

void moveMotors() {
  unsigned long now = micros();

  // Motor 1
  if (delay1_us > 0) {
    if (now - lastStep1 >= delay1_us) {
      lastStep1 = now;
      stepState1 = !stepState1;
      digitalWrite(STEP1, stepState1);
    }
  } else {
    digitalWrite(STEP1, LOW);
    stepState1 = LOW;
  }

  // Motor 2
  if (delay2_us > 0) {
    if (now - lastStep2 >= delay2_us) {
      lastStep2 = now;
      stepState2 = !stepState2;
      digitalWrite(STEP2, stepState2);
    }
  } else {
    digitalWrite(STEP2, LOW);
    stepState2 = LOW;
  }
}

void sendDataToPython() {
  unsigned long now = millis();

  if (now - lastSend >= SEND_INTERVAL) {
    lastSend = now;

    int rx = analogRead(JOY_X);
    int ry = analogRead(JOY_Y);

    float current_A = ina3221.getCurrent(INA_CHANNEL_1);
    float current_mA = current_A * 1000.0;

    Serial.print("DATA,");
    Serial.print(rx);
    Serial.print(",");
    Serial.print(ry);
    Serial.print(",");
    Serial.println(current_mA, 2);
  }
}