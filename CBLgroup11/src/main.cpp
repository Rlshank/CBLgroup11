#include <Arduino.h>
#include <ArduinoJson.h>

int xAxis = 35;
int yAxis = 34;
int button = 25;

void setup() {
  Serial.begin(115200);

}

void loop() {
  float x = analogRead(xAxis);
  float y = analogRead(yAxis);

  StaticJsonDocument<100> doc;

  JsonObject joystick = doc.createNestedObject("joystick");
  joystick["x"] = x;
  joystick["y"] = y;

  serializeJson(doc, Serial);
  Serial.println();
  
  delay(50);
  
  
}