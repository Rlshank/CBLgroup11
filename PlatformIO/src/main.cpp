#include <Arduino.h>
#include <ArduinoJson.h>

int xAxis = 32;
int yAxis = 33;

void setup() {
  Serial.begin(115200);
}

void loop() {
  int x = analogRead(xAxis);  // 0-4095
  int y = analogRead(yAxis);  // 0-4095

  StaticJsonDocument<100> doc;
  JsonObject joystick = doc.createNestedObject("joystick");
  joystick["x"] = x;
  joystick["y"] = y;

  serializeJson(doc, Serial);
  Serial.println();

  delay(100);
}