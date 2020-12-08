import cv2
import os
import numpy as np
from PIL import Image
import RPi.GPIO as GPIO
import time
import socket
import pyotp
import datetime


global user_name
user_name = ['','','','','']
global face_chk
face_chk = 0
global data
global confidata
confidata = ''
global count
count = 0
global securitylevel
securitylevel = 2

#Motor
STOP = 0
FORWARD = 1
BACKWARD = 2

CH1 = 0
HIGH = 1
LOW = 0

#Motor Pin
IN1 = 2 #Pin 3
IN2 = 3 #Pin 5
ENA = 4 #Pin 8

#KeyPad Pin
L1 = 5 
L2 = 6
L3 = 13
L4 = 19
C1 = 12
C2 = 16
C3 = 20
C4 = 21

#LED Pin
LED1 = 23 #RED LED
LED2 = 24 #GREEN LED

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

GPIO.setup(L1, GPIO.OUT)
GPIO.setup(L2, GPIO.OUT)
GPIO.setup(L3, GPIO.OUT)
GPIO.setup(L4, GPIO.OUT)

GPIO.setup(C1, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(C2, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(C3, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(C4, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

GPIO.setup(LED1, GPIO.OUT)
GPIO.setup(LED2, GPIO.OUT)

def camera(): # User Register
    global user_name
    
    vivi = cv2.VideoCapture(-1)
    vivi.set(3, 640)
    vivi.set(4, 480)
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    face_id = input('\n User ID(0~4): ')
    user_name[int(face_id)] = input('\n User Name: ')
    print('\n Save Face Start')
    
    count = 0

    while True:
        ret, img = vivi.read()
        
        if not ret:
            print('error')
            break
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1

            cv2.imwrite("./data/" + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        
        cv2.imshow('image', img)
            
        if count >= 50:
            break
        
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
        
    print('\n Save Face Finish')
    vivi.release()
    cv2.destroyAllWindows()


def getImagesAndLabels(path,detector):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[0])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
            
    return faceSamples,ids


def train(): #Face Training
    path = 'data'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
    print ("\n Training faces")
    
    faces,ids = getImagesAndLabels(path,detector)
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer.yml')
    
    print("\n Faces trained".format(len(np.unique(ids))))
    
def recog(): #User Detect
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
    font = cv2.FONT_HERSHEY_SIMPLEX

    id = 0

    global user_name
    global face_chk
    global securitylevel

    vivi = cv2.VideoCapture(-1)
    vivi.set(3, 640)
    vivi.set(4, 480)
    minW = 0.1*vivi.get(3)
    minH = 0.1*vivi.get(4)
    chkconfi = 0
    
    while True:
        ret, img = vivi.read()
        
        if not ret:
            print('error')
            break
        
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (int(minW), int(minH)), )
    
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            chkconfi = confidence
                
            if (confidence < 100):
                id = user_name[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                #confidence = "  {0}%".format(round(100 - confidence))
        
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
        cv2.imshow('camera', img)
        
        if (id == 'unknown'):
            print('Unknown User Detected!\n')
            break
        
        elif (chkconfi >= 40 and id != ''):
            print(id + ' Face OK')
            if securitylevel == 1:
                face_chk = 2
            else:
                face_chk = 1
            GPIO.output(23, False) #RED LED OFF
            GPIO.output(24, True)  #GREEN LED ON
            break

        k = cv2.waitKey(100) & 0xff
        
        if k == 27:
            break

    print("\n Exiting Program")
    vivi.release()
    cv2.destroyAllWindows()
    

def chkLine(line, characters):
    global confidata
    global count
    
    GPIO.output(line, GPIO.HIGH)
    
    if(GPIO.input(C1) == 1):
        confidata += (characters[0])
        count += 1
        print(confidata)
        
    if(GPIO.input(C2) == 1):
        confidata += (characters[1])
        count += 1
        print(confidata)
        
    if(GPIO.input(C3) == 1):
        confidata += (characters[2])
        count += 1
        print(confidata)
        
    if(GPIO.input(C4) == 1):
        confidata += (characters[3])
        count +=1
        print(confidata)
        
    GPIO.output(line, GPIO.LOW)


def user_otp(): #OTP Publish
    host = '192.168.0.11' 
    port = 8888

    server_sock = socket.socket(socket.AF_INET)
    server_sock.bind((host, port))
    server_sock.listen(1)

    print("OTP Check")
    client_sock, addr = server_sock.accept()
    print('Connected by', addr)


    totp = pyotp.TOTP('GAYDAMBQGAYDAMBQGAYDAMBQGA======')
    
    global data

    data = str(totp.now())
    client_sock.send(data.encode());

    print('OTP: ' + data)

    client_sock.close()
    server_sock.close() 
    

def chk_otp(): #Check OTP
    global data
    global face_chk
    global confidata
    global count
    
    while True:
        chkLine(L1, ["1","2","3","A"])
        chkLine(L2, ["4","5","6","B"])
        chkLine(L3, ["7","8","9","C"])
        chkLine(L4, ["*","0","#","D"])
        time.sleep(0.5)
        if count == 6:
            count = 0
            break
    
    if data == confidata:
        face_chk = 2
        confidata = ''
        count = 0
        
    confidata=''

def keyA(): # User Register, Face Train
    GPIO.output(L1, GPIO.HIGH)
    
    if(GPIO.input(C4) == 1):
        camera()
        train()
        
    GPIO.output(L1, GPIO.LOW)
    
def keyB(): # Face Detecting 
    GPIO.output(L2, GPIO.HIGH)
    
    if(GPIO.input(C4) == 1):
        recog()
        
    GPIO.output(L2, GPIO.LOW)

def keyC(): # Security Level
    GPIO.output(L3, GPIO.HIGH)
    global securitylevel
    change = 0
    if(GPIO.input(C4) == 1):
        print('Security Level Setting!')
        print('1:Face ID, 2:Face ID + OTP\n')
        change = 1
    
    if change == 1:
        while True:
            GPIO.output(L1, GPIO.HIGH)
        
            if(GPIO.input(C1) == 1):
                print('Security Level: 1(Face ID)')
                securitylevel = 1
                change = 0
                break
            
            elif(GPIO.input(C2) == 1):
                print('Security Level: 2(Face ID + OTP)')
                securitylevel = 2
                change = 0
                break
            
    GPIO.output(L3, GPIO.LOW)

def keyD(): # State 
    GPIO.output(L4, GPIO.HIGH)
    global securitylevel
    global user_name
    
    if(GPIO.input(C4) == 1):
        print('Security Level: ' + str(securitylevel))
        print('face_id      user_name')
        for i in range (0,5):
            print('   ' + str(i) + '          ' + str(user_name[i]))
            
    GPIO.output(L4, GPIO.LOW)
        
        
def setPinConfig(EN, INA, INB):
	GPIO.setup(EN, GPIO.OUT)
	GPIO.setup(INA, GPIO.OUT)
	GPIO.setup(INB, GPIO.OUT)
	pwm = GPIO.PWM(EN, 100)
	pwm.start(0)
	return pwm

def setMotorControl(pwm, INA, INB, speed, stat):
	pwm.ChangeDutyCycle(speed)
	if(stat == FORWARD):
		GPIO.output(INA, HIGH)
		GPIO.output(INB, LOW)
	elif(stat == BACKWARD):
		GPIO.output(INA, LOW)
		GPIO.output(INB, HIGH)
	elif(stat == STOP):
		GPIO.output(INA, LOW)
		GPIO.output(INB, LOW)

def setMotor(ch, speed, stat):
	if(ch == CH1):
		setMotorControl(pwmA, IN1, IN2, speed, stat)

GPIO.setmode(GPIO.BCM)
pwmA = setPinConfig(ENA, IN1, IN2)

print('Digital Doorlock \n')
print('A:User Register B:Face Detecting C:Security Level D:State \n')

while True:
    GPIO.output(23, True) #RED LED ON
    GPIO.output(24, False) #GREEN LED OFF
    
    keyA()  # User, Face Train
    keyB()  # Detect
    keyC()  # Security Level
    keyD()  # State
    
    if face_chk == 1:
        user_otp()
        chk_otp()
        
    if face_chk == 2:
        setMotor(CH1, 100, BACKWARD)
        print('Door Open!')
        time.sleep(3)
        setMotor(CH1, 100, STOP)
        time.sleep(3)
        setMotor(CH1, 100, FORWARD)
        print('Door Close!')
        time.sleep(3)
        setMotor(CH1, 100, STOP)
        face_chk = 0
    
    time.sleep(0.5)
