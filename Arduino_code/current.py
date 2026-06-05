#include <Wire.h>

// INA3221 I2C address (A0 pin to GND = 0x40)
#define INA3221_ADDR 0x40

// INA3221 registers
#define REG_SHUNT_V_CH1 0x01
#define REG_CONFIG      0x00

const float SHUNT_R = 0.1;  // your shunt resistor in ohms
const float Kt = 0.24;      // replace with your measured Kt (Nm/A)

void setup() {
  Serial.begin(115200);
  Wire.begin();
  
  // Reset and configure INA3221
  writeRegister(REG_CONFIG, 0x7127); // all 3 channels on, 1024 averages
  
  Serial.println("I2C Address scan:");
  scanI2C(); // confirm INA3221 is detected
  
  Serial.println("Current (A) | Torque (Nm)");
}

void loop() {
  float current = readCurrent();
  float torque = Kt * abs(current);
  
  Serial.print(current, 4);
  Serial.print(" A  |  ");
  Serial.print(torque, 4);
  Serial.println(" Nm");
  
  delay(100);
}

// ---- INA3221 functions ----

float readCurrent() {
  int16_t raw = readRegister(REG_SHUNT_V_CH1);
  float shuntVoltage = raw * 40e-6; // LSB = 40uV
  return shuntVoltage / SHUNT_R;
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
  Wire.endTransmission(false);
  Wire.requestFrom(INA3221_ADDR, 2);
  int16_t value = (Wire.read() << 8) | Wire.read();
  return value >> 3; // INA3221 shunt register is 13-bit, shift by 3
}

// Scan to confirm INA3221 is visible on I2C bus
void scanI2C() {
  for (byte addr = 8; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.print("  Found device at 0x");
      Serial.println(addr, HEX);
    }
  }
}