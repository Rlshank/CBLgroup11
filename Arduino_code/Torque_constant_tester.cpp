// =============================================
// NEMA 17 Torque Constant (Kt) Finder
// Spool + hanging weight method
// =============================================
// Wiring:
//   - Stepper driver STEP  -> pin 3
//   - Stepper driver DIR   -> pin 4
//   - Stepper driver ENABLE-> pin 5 (LOW = enabled)
//   - Current sense resistor (e.g. 0.1 ohm) between
//     motor GND and Arduino A0 (with op-amp if needed)
//   - Or: just enter current manually via Serial
// =============================================

#include <AccelStepper.h>

#define STEP_PIN   3
#define DIR_PIN    4
#define EN_PIN     5
#define CURRENT_PIN A0

AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

// ---- Config — edit these ----
const float SPOOL_RADIUS_M  = 0.0025;  // metres (bare shaft = 0.0025)
const float GRAVITY         = 9.81;    // m/s²
const int   NUM_TRIALS      = 5;       // how many mass/current combos
const bool  MANUAL_CURRENT  = true;    // true = you type current via Serial
                                       // false = read from A0 (needs sense circuit)
const float SHUNT_OHMS      = 0.1;     // only used if MANUAL_CURRENT = false
const float AMP_GAIN        = 10.0;    // op-amp gain on shunt, if any
// -----------------------------

float masses_kg[10];
float currents_A[10];
float torques_Nm[10];
float kt_values[10];
int   trialCount = 0;

float readCurrent() {
  if (MANUAL_CURRENT) return -1;
  int raw = analogRead(CURRENT_PIN);
  float voltage = raw * (5.0 / 1023.0);
  return (voltage / AMP_GAIN) / SHUNT_OHMS;
}

float waitForFloat(const char* prompt) {
  Serial.print(prompt);
  while (!Serial.available());
  float val = Serial.parseFloat();
  Serial.println(val);
  return val;
}

void printResults() {
  Serial.println(F("\n========================================"));
  Serial.println(F("  RESULTS"));
  Serial.println(F("========================================"));
  Serial.println(F(" #  Mass(kg)  I(A)   T(Nm)    Kt(Nm/A)"));
  Serial.println(F("----------------------------------------"));

  float kt_sum = 0, kt_min = kt_values[0], kt_max = kt_values[0];

  for (int i = 0; i < trialCount; i++) {
    char buf[50];
    snprintf(buf, sizeof(buf), " %d  %.4f   %.3f  %.5f  %.4f",
             i + 1, masses_kg[i], currents_A[i], torques_Nm[i], kt_values[i]);
    Serial.println(buf);
    kt_sum += kt_values[i];
    if (kt_values[i] < kt_min) kt_min = kt_values[i];
    if (kt_values[i] > kt_max) kt_max = kt_values[i];
  }

  float kt_avg = kt_sum / trialCount;

  // std deviation
  float variance = 0;
  for (int i = 0; i < trialCount; i++)
    variance += pow(kt_values[i] - kt_avg, 2);
  float kt_std = sqrt(variance / trialCount);

  // R² (torque vs current, forced through origin)
  float ss_res = 0, ss_tot = 0;
  float t_mean = 0;
  for (int i = 0; i < trialCount; i++) t_mean += torques_Nm[i];
  t_mean /= trialCount;
  for (int i = 0; i < trialCount; i++) {
    float predicted = kt_avg * currents_A[i];
    ss_res += pow(torques_Nm[i] - predicted, 2);
    ss_tot += pow(torques_Nm[i] - t_mean, 2);
  }
  float r2 = (ss_tot > 0) ? (1.0 - ss_res / ss_tot) : 1.0;

  Serial.println(F("----------------------------------------"));
  Serial.print(F("  Average Kt : ")); Serial.print(kt_avg, 4); Serial.println(F(" N.m/A"));
  Serial.print(F("  Std dev    : ")); Serial.print(kt_std, 4); Serial.println(F(" N.m/A"));
  Serial.print(F("  Min Kt     : ")); Serial.print(kt_min, 4); Serial.println(F(" N.m/A"));
  Serial.print(F("  Max Kt     : ")); Serial.print(kt_max, 4); Serial.println(F(" N.m/A"));
  Serial.print(F("  R squared  : ")); Serial.println(r2, 4);

  if (r2 >= 0.99)      Serial.println(F("  Linearity  : Excellent"));
  else if (r2 >= 0.95) Serial.println(F("  Linearity  : Acceptable"));
  else                 Serial.println(F("  Linearity  : Poor - recheck setup"));

  Serial.println(F("========================================\n"));
}

void setup() {
  Serial.begin(115200);
  pinMode(EN_PIN, OUTPUT);
  digitalWrite(EN_PIN, LOW);  // enable driver

  stepper.setMaxSpeed(0);     // hold only, no movement needed
  stepper.setAcceleration(0);

  Serial.println(F("\n========================================"));
  Serial.println(F("  NEMA 17 Kt Measurement Tool"));
  Serial.println(F("========================================"));
  Serial.print(F("  Spool radius : ")); Serial.print(SPOOL_RADIUS_M * 1000, 2); Serial.println(F(" mm"));
  Serial.print(F("  Trials       : ")); Serial.println(NUM_TRIALS);
  Serial.println(F("\n  Set your driver to hold mode (no stepping)."));
  Serial.println(F("  For each trial: set current on driver, hang"));
  Serial.println(F("  the weight that just causes slip, then enter"));
  Serial.println(F("  the values below.\n"));
}

void loop() {
  if (trialCount < NUM_TRIALS) {
    Serial.print(F("\n--- Trial "));
    Serial.print(trialCount + 1);
    Serial.print(F(" of "));
    Serial.println(NUM_TRIALS);

    float mass = waitForFloat("  Hanging mass (kg): ");

    float current;
    if (MANUAL_CURRENT) {
      current = waitForFloat("  Winding current (A): ");
    } else {
      current = readCurrent();
      Serial.print(F("  Measured current (A): "));
      Serial.println(current, 3);
    }

    float force   = mass * GRAVITY;
    float torque  = force * SPOOL_RADIUS_M;
    float kt      = (current > 0) ? torque / current : 0;

    masses_kg[trialCount]  = mass;
    currents_A[trialCount] = current;
    torques_Nm[trialCount] = torque;
    kt_values[trialCount]  = kt;

    Serial.print(F("  Force  : ")); Serial.print(force, 4);  Serial.println(F(" N"));
    Serial.print(F("  Torque : ")); Serial.print(torque, 5); Serial.println(F(" N.m"));
    Serial.print(F("  Kt     : ")); Serial.print(kt, 4);     Serial.println(F(" N.m/A"));

    trialCount++;

    if (trialCount == NUM_TRIALS) {
      printResults();
      Serial.println(F("Done. Reset Arduino to run again."));
    }
  }
}