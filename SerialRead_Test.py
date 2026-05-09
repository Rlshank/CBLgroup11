import serial
import json

ser = serial.Serial('COM16', 115200)  
while True:
    line = ser.readline()

    data = json.loads(line)
    x = data["joystick"]["x"]
    y = data["joystick"]["y"]
    print(f"X: {x}, Y: {y}")


