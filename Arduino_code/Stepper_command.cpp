#define STEP1  3
#define DIR1   7
#define STEP2  5
#define DIR2   4
#define VRx    A0
#define VRy    A1

int targetSpeed1 = 0;
int targetSpeed2 = 0;
bool dir1 = true;
bool dir2 = true;

unsigned long lastStep1 = 0;
unsigned long lastStep2 = 0;
bool stepState1 = false;
bool stepState2 = false;

void setup() {
  pinMode(STEP1, OUTPUT);
  pinMode(DIR1,  OUTPUT);
  pinMode(STEP2, OUTPUT);
  pinMode(DIR2,  OUTPUT);
  Serial.begin(115200);
}

void loop() {
  unsigned long now = micros();

  // Non-blocking serial read
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    int d1, sp1, d2, sp2;
    if (sscanf(cmd.c_str(), "%d,%d,%d,%d", &d1, &sp1, &d2, &sp2) == 4) {
      dir1 = d1;
      dir2 = d2;
      targetSpeed1 = sp1;
      targetSpeed2 = sp2;
      digitalWrite(DIR1, dir1 ? HIGH : LOW);
      digitalWrite(DIR2, dir2 ? HIGH : LOW);
    }
  }

  // Motor 1 - non-blocking
  if (targetSpeed1 > 0 && (now - lastStep1 >= (unsigned long)targetSpeed1)) {
    stepState1 = !stepState1;
    digitalWrite(STEP1, stepState1 ? HIGH : LOW);
    lastStep1 = now;
  }

  // Motor 2 - non-blocking
  if (targetSpeed2 > 0 && (now - lastStep2 >= (unsigned long)targetSpeed2)) {
    stepState2 = !stepState2;
    digitalWrite(STEP2, stepState2 ? HIGH : LOW);
    lastStep2 = now;
  }

  // Send joystick every ~50ms
  static unsigned long lastSend = 0;
  if (millis() - lastSend >= 50) {
    Serial.print(analogRead(VRx));
    Serial.print(",");
    Serial.println(analogRead(VRy));
    lastSend = millis();
  }
}