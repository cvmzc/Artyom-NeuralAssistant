import RPi.GPIO as GPIO #Importe la bibliothèque pour contrôler les GPIOs
import time

GPIO.setmode(GPIO.BOARD) #Définit le mode de numérotation (Board)
GPIO.setwarnings(False)

Red = 7
Yellow = 12
Green = 18

def LED_Red():
    GPIO.setup(Red, GPIO.OUT)
    if GPIO.input(Red) == GPIO.LOW:
        GPIO.output(Red, GPIO.HIGH)
    else:
        GPIO.output(Red, GPIO.LOW)

def LED_Yellow():
    GPIO.setup(Yellow, GPIO.OUT)
    if GPIO.input(Yellow) == GPIO.LOW:
        GPIO.output(Yellow, GPIO.HIGH)
    else:
        GPIO.output(Yellow, GPIO.LOW)

def LED_Green():
    GPIO.setup(Green, GPIO.OUT)
    if GPIO.input(Green) == GPIO.LOW:
        GPIO.output(Green, GPIO.HIGH)
    else:
        GPIO.output(Green, GPIO.LOW)

def Clean():
    GPIO.cleanup()