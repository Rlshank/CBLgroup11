#include <Wire.h>

// INA3221 I2C address
#define INA3221_ADDR 0x40

// INA3221 registers
#define REG_CONFIG       0x00
#define REG_SHUNT_V_CH1  0x01   // CH1
// #define REG_SHUNT_V_CH2  0x03 // CH2
// #define REG_SHUNT_V_CH3  0x05 // CH3

const float SHUNT_R = 0.1;  // R100 = 0.1 ohm
const float Kt = 0.24;      // Nm/A

void setup() {
  Serial.begin(115200);
  Wire.begin();

  delay(500);

  Serial.println("I2C Address scan:");
  scanI2C();

  writeRegister(REG_CONFIG, 0x7127);

  Serial.println("Current CH1 (A) | Torque indication (Nm)");
}

void loop() {
  float current = readCurrent();
  float torque = Kt * abs(current);

  Serial.print(current, 4);
  Serial.print(" A  |  ");
  Serial.print(torque, 4);
  Serial.println(" Nm");

  delay(200);
}

float readCurrent() {
  int16_t raw = readRegister(REG_SHUNT_V_CH1);

  float shuntVoltage = raw * 40e-6; // 40 microvolt per bit
  float current = shuntVoltage / SHUNT_R;

  return current;
}

void writeRegister(uint8_t reg, uint16_t value) {
  Wire.beginTransmission(INA3221_ADDR);
  Wire.write(reg);
  Wire.write((value >> 8) & 0xFF);
  Wire.write(value & 0xFF);
  Wire.endTransmission();
}

int16_t readRegister(uint8_t reg) {
  Wire.beginTransmission(INA3221_ADDR);
  Wire.write(reg);

  if (Wire.endTransmission(false) != 0) {
    Serial.println("I2C read error");
    return 0;
  }

  Wire.requestFrom(INA3221_ADDR, (uint8_t)2);

  if (Wire.available() < 2) {
    Serial.println("Not enough data received");
    return 0;
  }

  int16_t value = (Wire.read() << 8) | Wire.read();

  return value >> 3;
}

void scanI2C() {
  for (byte addr = 8; addr < 127; addr++) {
    Wire.beginTransmission(addr);

    if (Wire.endTransmission() == 0) {
      Serial.print("Found device at 0x");
      Serial.println(addr, HEX);
    }
  }
}