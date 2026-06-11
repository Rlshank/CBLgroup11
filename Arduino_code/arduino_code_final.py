#include <math.h>

// ===================================================
// Smooth inverse velocity kinematics
// 2-cable system, 93 cm wide, 50 cm high
//
// Coordinate system:
// x = -46.5 cm to +46.5 cm
// y = 0 cm to 50 cm
//
// Start position:
// x = 0, y = 0
// ===================================================

// =========================
// Joystick pins
// =========================
#define PIN_A0 A0
#define PIN_A1 A1

// =========================
// Motor pins
// =========================
// Motor 1 = left motor
#define STEP1 8
#define DIR1 7

// Motor 2 = right motor
#define STEP2 3
#define DIR2 4

// =========================
// Motor direction settings
// =========================
// These are your current working direction settings
#define M1_WIND_IN HIGH
#define M1_UNWIND  LOW

#define M2_WIND_IN LOW
#define M2_UNWIND  HIGH

// =========================
// Geometry in meters
// =========================
const float WIDTH = 0.93;
const float HEIGHT = 0.50;
const float HALF_WIDTH = WIDTH / 2.0;

const float LEFT_X = -HALF_WIDTH;
const float LEFT_Y = HEIGHT;

const float RIGHT_X = HALF_WIDTH;
const float RIGHT_Y = HEIGHT;

// Estimated camera/weight position
float camX = 0.0;
float camY = 0.0;

// Movement limits
const float MIN_X = -HALF_WIDTH;
const float MAX_X = HALF_WIDTH;
const float MIN_Y = 0.0;
const float MAX_Y = HEIGHT;

// Set to false so Arduino does NOT stop at the edges.
// Later the digital twin can block movement based on tension/current.
const bool USE_POSITION_LIMITS = false;

// =========================
// Motor / spool settings
// =========================
const float SPOOL_RADIUS = 0.0125;   // 1.25 cm
const int STEPS_PER_REV = 200;       // full step because MS pins are not connected

const float CABLE_PER_STEP = (2.0 * PI * SPOOL_RADIUS) / STEPS_PER_REV;

// =========================
// Control settings
// =========================
const int DEADZONE = 200;

// Movement speed in m/s
// 0.10 = 10 cm/s
const float MOVE_SPEED = 0.15;

// Higher value = slower but smoother/less skipping
const unsigned long MIN_STEP_INTERVAL_US = 1200;

// =========================
// Joystick center values
// =========================
int centerA0 = 512;
int centerA1 = 512;

// =========================
// Step timing
// =========================
unsigned long lastStep1 = 0;
unsigned long lastStep2 = 0;

unsigned long interval1 = 999999;
unsigned long interval2 = 999999;

bool motor1Active = false;
bool motor2Active = false;

unsigned long lastLoopTime = 0;

// =========================
// Functions
// =========================
float ropeLength(float x, float y, float anchorX, float anchorY) {
  float dx = x - anchorX;
  float dy = y - anchorY;
  return sqrt(dx * dx + dy * dy);
}

void stepMotor(int stepPin) {
  digitalWrite(stepPin, HIGH);
  delayMicroseconds(5);
  digitalWrite(stepPin, LOW);
}

// =========================
// Setup
// =========================
void setup() {
  Serial.begin(115200);

  pinMode(STEP1, OUTPUT);
  pinMode(DIR1, OUTPUT);

  pinMode(STEP2, OUTPUT);
  pinMode(DIR2, OUTPUT);

  digitalWrite(STEP1, LOW);
  digitalWrite(STEP2, LOW);

  // Calibrate joystick center
  // Do not touch the joystick during startup
  long sumA0 = 0;
  long sumA1 = 0;

  for (int i = 0; i < 100; i++) {
    sumA0 += analogRead(PIN_A0);
    sumA1 += analogRead(PIN_A1);
    delay(5);
  }

  centerA0 = sumA0 / 100;
  centerA1 = sumA1 / 100;

  lastLoopTime = micros();

  Serial.println("Arduino inverse kinematics started");
  Serial.print("Center A0 = ");
  Serial.println(centerA0);
  Serial.print("Center A1 = ");
  Serial.println(centerA1);
  Serial.println("Start position: X = 0 cm, Y = 0 cm");

  if (USE_POSITION_LIMITS) {
    Serial.println("Position limits: ON");
  } else {
    Serial.println("Position limits: OFF");
  }
}

// =========================
// Main loop
// =========================
void loop() {
  unsigned long now = micros();

  float dt = (now - lastLoopTime) / 1000000.0;
  lastLoopTime = now;

  // =========================
  // Read joystick
  // =========================
  int rawA0 = analogRead(PIN_A0) - centerA0;
  int rawA1 = analogRead(PIN_A1) - centerA1;

  // Correct joystick mapping from your tests
  int xValue = -rawA0;
  int yValue = -rawA1;

  float vx = 0.0;
  float vy = 0.0;

  // Only allow one direction at a time
  if (abs(xValue) > abs(yValue) && abs(xValue) > DEADZONE) {
    if (xValue > 0) {
      vx = MOVE_SPEED;      // right
    } else {
      vx = -MOVE_SPEED;     // left
    }
  } 
  else if (abs(yValue) > DEADZONE) {
    if (yValue > 0) {
      vy = MOVE_SPEED;      // up
    } else {
      vy = -MOVE_SPEED;     // down
    }
  }

  // =========================
  // Optional software limits
  // =========================
  if (USE_POSITION_LIMITS) {
    if (camX <= MIN_X && vx < 0) vx = 0;
    if (camX >= MAX_X && vx > 0) vx = 0;
    if (camY <= MIN_Y && vy < 0) vy = 0;
    if (camY >= MAX_Y && vy > 0) vy = 0;
  }

  // =========================
  // Inverse velocity kinematics
  // =========================
  float L1 = ropeLength(camX, camY, LEFT_X, LEFT_Y);
  float L2 = ropeLength(camX, camY, RIGHT_X, RIGHT_Y);

  // Cable speed in m/s
  float dL1dt = ((camX - LEFT_X) * vx + (camY - LEFT_Y) * vy) / L1;
  float dL2dt = ((camX - RIGHT_X) * vx + (camY - RIGHT_Y) * vy) / L2;

  // Motor 1 direction
  if (dL1dt < 0) {
    digitalWrite(DIR1, M1_WIND_IN);
  } else {
    digitalWrite(DIR1, M1_UNWIND);
  }

  // Motor 2 direction
  if (dL2dt < 0) {
    digitalWrite(DIR2, M2_WIND_IN);
  } else {
    digitalWrite(DIR2, M2_UNWIND);
  }

  // Convert cable speed to step frequency
  float freq1 = fabs(dL1dt) / CABLE_PER_STEP;
  float freq2 = fabs(dL2dt) / CABLE_PER_STEP;

  if (freq1 > 1.0) {
    interval1 = 1000000.0 / freq1;
    if (interval1 < MIN_STEP_INTERVAL_US) interval1 = MIN_STEP_INTERVAL_US;
    motor1Active = true;
  } else {
    motor1Active = false;
  }

  if (freq2 > 1.0) {
    interval2 = 1000000.0 / freq2;
    if (interval2 < MIN_STEP_INTERVAL_US) interval2 = MIN_STEP_INTERVAL_US;
    motor2Active = true;
  } else {
    motor2Active = false;
  }

  // =========================
  // Generate step pulses
  // =========================
  now = micros();

  if (motor1Active && now - lastStep1 >= interval1) {
    lastStep1 = now;
    stepMotor(STEP1);
  }

  if (motor2Active && now - lastStep2 >= interval2) {
    lastStep2 = now;
    stepMotor(STEP2);
  }

  // =========================
  // Update estimated position
  // =========================
  camX += vx * dt;
  camY += vy * dt;

  // Only constrain position if limits are enabled
  if (USE_POSITION_LIMITS) {
    camX = constrain(camX, MIN_X, MAX_X);
    camY = constrain(camY, MIN_Y, MAX_Y);
  }

  // =========================
  // Send data to Python
  // Format:
  // POS,x,y,joystick_x,joystick_y,current_mA
  // =========================
  static unsigned long lastSend = 0;
  if (millis() - lastSend > 50) {
    lastSend = millis();

    Serial.print("POS,");
    Serial.print(camX, 4);
    Serial.print(",");
    Serial.print(camY, 4);
    Serial.print(",");
    Serial.print(xValue);
    Serial.print(",");
    Serial.print(yValue);
    Serial.print(",");
    Serial.println(0.0);   // Current placeholder for now
  }
}