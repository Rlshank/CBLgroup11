#include <math.h>
#include "HX711.h"

// ===================================================
// Corrected inverse kinematics code
// 2-cable system with 2 HX711 load cells
//
// Twin receives:
// POS,x,y,joystick_x,joystick_y,loadcell1,loadcell2
//
// loadcell1 and loadcell2 are in Newtons
// ===================================================

// =========================
// Joystick pins
// =========================
#define PIN_A0 A0   // joystick X
#define PIN_A1 A1   // joystick Y

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
// Load cell pins
// =========================
#define LOADCELL1_DOUT 5
#define LOADCELL1_SCK  6

#define LOADCELL2_DOUT 11
#define LOADCELL2_SCK  12

HX711 loadCell1;
HX711 loadCell2;

// Calibration factor for both load cells
const float LOADCELL_CALIBRATION_FACTOR = 2066.1638;

// If your calibration was done with grams, keep this true
const bool LOADCELL_OUTPUT_IS_GRAMS = true;

const float GRAVITY = 9.81;

// Force values in Newtons
float force1_N = 0.0;
float force2_N = 0.0;

// =========================
// Correct motor direction settings
// =========================
#define M1_WIND_IN LOW
#define M1_UNWIND  HIGH

#define M2_WIND_IN HIGH
#define M2_UNWIND  LOW

// =========================
// Geometry in meters
// =========================
const float WIDTH = 0.93;
const float HEIGHT = 0.505;
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

// Left/right/bottom limits ON
const bool USE_POSITION_LIMITS = true;

// Top limit OFF, so the twin can decide when to block
const bool USE_TOP_LIMIT = false;

// Prevent IK math from flipping or stopping near/above the top
const bool PREVENT_TOP_IK_FLIP = true;
const float IK_TOP_MARGIN = 0.08;

// =========================
// Movement mode
// =========================
// true  = diagonal movement allowed
// false = only left/right OR up/down
const bool ALLOW_DIAGONAL_MOVEMENT = false;

// =========================
// Motor / spool settings
// =========================
const float SPOOL_RADIUS = 0.0125;
const int STEPS_PER_REV = 200;

const float CABLE_PER_STEP = (2.0 * PI * SPOOL_RADIUS) / STEPS_PER_REV;

// =========================
// Control settings
// =========================
const int DEADZONE = 120;
const float MOVE_SPEED = 0.15;
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
// Helper functions
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

float clampFloat(float value, float minValue, float maxValue) {
  if (value < minValue) return minValue;
  if (value > maxValue) return maxValue;
  return value;
}

float joystickToUnit(int raw) {
  if (abs(raw) <= DEADZONE) {
    return 0.0;
  }

  float sign = raw > 0 ? 1.0 : -1.0;
  float value = (abs(raw) - DEADZONE) / (512.0 - DEADZONE);

  value = clampFloat(value, 0.0, 1.0);

  return sign * value;
}

float loadCellValueToNewton(float value) {
  if (LOADCELL_OUTPUT_IS_GRAMS) {
    return (value / 1000.0) * GRAVITY;
  } else {
    return value;
  }
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

  // =========================
  // Setup load cells
  // =========================
  loadCell1.begin(LOADCELL1_DOUT, LOADCELL1_SCK);
  loadCell2.begin(LOADCELL2_DOUT, LOADCELL2_SCK);

  loadCell1.set_scale(LOADCELL_CALIBRATION_FACTOR);
  loadCell2.set_scale(LOADCELL_CALIBRATION_FACTOR);

  // Important:
  // During tare, the load cells should be unloaded or at the chosen zero-tension state.
  Serial.println("Taring load cells...");
  loadCell1.tare();
  loadCell2.tare();
  Serial.println("Load cells tared.");

  // =========================
  // Calibrate joystick center
  // =========================
  Serial.println("Do not touch joystick. Calibrating...");

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

  Serial.println("Corrected inverse kinematics with load cells started");
  Serial.print("Center A0 = ");
  Serial.println(centerA0);
  Serial.print("Center A1 = ");
  Serial.println(centerA1);

  Serial.println("Output format:");
  Serial.println("POS,x,y,joystick_x,joystick_y,loadcell1,loadcell2");
}

// =========================
// Main loop
// =========================
void loop() {
  unsigned long now = micros();

  float dt = (now - lastLoopTime) / 1000000.0;
  lastLoopTime = now;
  
  if (dt > 0.1) {
    dt = 0.1;
  }

  static float joystickScale = 1.0;

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
  if (cmd.startsWith("S")) {
    joystickScale = cmd.substring(1).toFloat();
  }
  }

  // =========================
  // Read joystick
  // =========================
  int rawA0 = analogRead(PIN_A0) - centerA0;
  int rawA1 = analogRead(PIN_A1) - centerA1;

  float xJoy = joystickToUnit(rawA0);
  float yJoy = joystickToUnit(rawA1);

  // =========================
  // Diagonal movement setting
  // =========================
  if (ALLOW_DIAGONAL_MOVEMENT) {
    float joyLength = sqrt(xJoy * xJoy + yJoy * yJoy);

    if (joyLength > 1.0) {
      xJoy = xJoy / joyLength;
      yJoy = yJoy / joyLength;
    }
  } else {
    if (fabs(xJoy) > fabs(yJoy)) {
      yJoy = 0.0;
    } else {
      xJoy = 0.0;
    }
  }

  // Desired camera velocity
  float vx = xJoy * MOVE_SPEED *joystickScale;
  float vy = yJoy * MOVE_SPEED *joystickScale;

  // =========================
  // Software position limits
  // =========================
  if (USE_POSITION_LIMITS) {
    if (camX <= MIN_X && vx < 0) {
      vx = 0;
    }

    if (camX >= MAX_X && vx > 0) {
      vx = 0;
    }

    if (camY <= MIN_Y && vy < 0) {
      vy = 0;
    }

    if (USE_TOP_LIMIT) {
      if (camY >= MAX_Y && vy > 0) {
        vy = 0;
      }
    }
  }

  // =========================
  // Inverse velocity kinematics
  // =========================
  float ikX = camX;
  float ikY = camY;

  if (PREVENT_TOP_IK_FLIP && ikY >= HEIGHT - IK_TOP_MARGIN) {
    ikY = HEIGHT - IK_TOP_MARGIN;
  }

  float L1 = ropeLength(ikX, ikY, LEFT_X, LEFT_Y);
  float L2 = ropeLength(ikX, ikY, RIGHT_X, RIGHT_Y);

  float dL1dt = 0.0;
  float dL2dt = 0.0;

  if (L1 > 0.001) {
    dL1dt = ((ikX - LEFT_X) * vx + (ikY - LEFT_Y) * vy) / L1;
  }

  if (L2 > 0.001) {
    dL2dt = ((ikX - RIGHT_X) * vx + (ikY - RIGHT_Y) * vy) / L2;
  }

  // dL/dt < 0 means cable gets shorter = wind in
  // dL/dt > 0 means cable gets longer  = unwind

  if (dL1dt < 0) {
    digitalWrite(DIR1, M1_WIND_IN);
  } else {
    digitalWrite(DIR1, M1_UNWIND);
  }

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

    if (interval1 < MIN_STEP_INTERVAL_US) {
      interval1 = MIN_STEP_INTERVAL_US;
    }

    motor1Active = true;
  } else {
    motor1Active = false;
  }

  if (freq2 > 1.0) {
    interval2 = 1000000.0 / freq2;

    if (interval2 < MIN_STEP_INTERVAL_US) {
      interval2 = MIN_STEP_INTERVAL_US;
    }

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

  if (USE_POSITION_LIMITS) {
    camX = clampFloat(camX, MIN_X, MAX_X);

    if (camY < MIN_Y) {
      camY = MIN_Y;
    }

    if (USE_TOP_LIMIT) {
      camY = clampFloat(camY, MIN_Y, MAX_Y);
    }
  }

  // =========================
  // Read load cells and send data
  //
  // Format:
  // POS,x,y,joystick_x,joystick_y,loadcell1,loadcell2
  // =========================
  static unsigned long lastSend = 0;

  if (millis() - lastSend > 50) {
    lastSend = millis();

    if (loadCell1.is_ready()) {
      float rawValue1 = loadCell1.get_units(1);
      force1_N = loadCellValueToNewton(rawValue1);
    }

    if (loadCell2.is_ready()) {
      float rawValue2 = loadCell2.get_units(1);
      force2_N = loadCellValueToNewton(rawValue2);
    }

    Serial.print("POS,");
    Serial.print(camX, 4);
    Serial.print(",");
    Serial.print(camY, 4);
    Serial.print(",");
    Serial.print(xJoy, 3);
    Serial.print(",");
    Serial.print(yJoy, 3);
    Serial.print(",");
    Serial.print(force1_N, 3);
    Serial.print(",");
    Serial.println(force2_N, 3);
  }
}