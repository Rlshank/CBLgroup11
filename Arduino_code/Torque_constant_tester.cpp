// =============================================
// NEMA17 Kt Measurement (SIMPLIFIED)
// A4988 holding torque method
// =============================================

#define EN_PIN 5

const float SPOOL_RADIUS_M = 0.0025;
const float GRAVITY = 9.81;
const int NUM_TRIALS = 5;

float masses[NUM_TRIALS];
float torques[NUM_TRIALS];
float kt_values[NUM_TRIALS];

int trial = 0;

// ---- helper ----
float waitForFloat(const char* msg) {
  Serial.print(msg);
  while (!Serial.available());
  float v = Serial.parseFloat();
  Serial.println(v);
  return v;
}

void setup() {
  Serial.begin(115200);

  pinMode(EN_PIN, OUTPUT);
  digitalWrite(EN_PIN, LOW); // ENABLE A4988 (important!)

  Serial.println("\n=== Kt Measurement Tool ===");
  Serial.println("A4988 holding mode active");
  Serial.print("Spool radius: ");
  Serial.print(SPOOL_RADIUS_M * 1000);
  Serial.println(" mm\n");
}

void loop() {

  if (trial < NUM_TRIALS) {

    Serial.print("\n--- Trial ");
    Serial.print(trial + 1);
    Serial.println(" ---");

    float mass = waitForFloat("Mass (kg): ");

    float force = mass * GRAVITY;
    float torque = force * SPOOL_RADIUS_M;

    // FIXED CURRENT (from your Vref = 0.6V ≈ 1.5A)
    float current = 1.5;
    float kt = torque / current;

    masses[trial] = mass;
    torques[trial] = torque;
    kt_values[trial] = kt;

    Serial.print("Force  (N): ");
    Serial.println(force, 5);

    Serial.print("Torque (Nm): ");
    Serial.println(torque, 6);

    Serial.print("Kt     (Nm/A): ");
    Serial.println(kt, 6);

    trial++;
  }

  else {
    // ---- RESULTS ----
    Serial.println("\n================ RESULTS ================");

    float sum = 0;
    float minKt = kt_values[0];
    float maxKt = kt_values[0];

    for (int i = 0; i < NUM_TRIALS; i++) {
      sum += kt_values[i];
      if (kt_values[i] < minKt) minKt = kt_values[i];
      if (kt_values[i] > maxKt) maxKt = kt_values[i];

      Serial.print("Trial ");
      Serial.print(i + 1);
      Serial.print(": Kt = ");
      Serial.println(kt_values[i], 6);
    }

    float avg = sum / NUM_TRIALS;

    Serial.println("--------------------------------");
    Serial.print("Average Kt: ");
    Serial.println(avg, 6);

    Serial.print("Min Kt: ");
    Serial.println(minKt, 6);

    Serial.print("Max Kt: ");
    Serial.println(maxKt, 6);

    Serial.println("================================");

    while (1); // stop
  }
}