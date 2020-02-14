import cv2 
import csv
import os
import tkinter as t
from tkinter import messagebox as msg
import pandas as pd
import numpy as np
from PIL import Image
import time
import datetime
userent=rolent=nament=''
pasent=clr=clr1=clr2=clr3=lab=r=n=0
def verify():
    global r,n
    r=rolent.get()
    n=nament.get()
    
    if len(r)==10 and len(n)>=1:
        #print("10")
        msg.showinfo("info","Verification Successfull")
        enr=t.Button(home,text="ENROLL FACE",command=enroll)
        enr.place(x=20,y=200)
    elif len(n)== 0:
        msg.showwarning("warning","Please enter your name")
    else:        
        msg.showwarning("warning","Wrong Roll Number")    
        
def enroll():
        global n,r
        cam=cv2.VideoCapture(0)
        harcascadePath = 'haarcascade_frontalface_default.xml'
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret,img=cam.read()
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces=detector.detectMultiScale(gray,1.3,5)
            for(x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                sampleNum=sampleNum+1
                cv2.imwrite("TrainingImage\ "+n+"."+r +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                cv2.imshow('Frame',img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum > 15:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Saved for ID: "+r + "Name: "+n
        row = [r,n]
        with open('StudentDetails\studentDetails.csv', 'a+') as csvFile:
            writer=csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        msg.showinfo("Registered Successfully","Name:"+n+"\nRoll NUmber:"+r)
        
    

def train():
    recognizer = cv2.face_LBPHFaceRecognizer.create()#recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"#+",".join(str(f) for f in Id)
    msg.showinfo("info",res)


def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #print(imagePaths)
    
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids



def submit():
    pass

def clear():
    userent.delete(first=0,last=len(userent.get()))
                   
def clear1():
    pasent.delete(first=0,last=len(pasent.get()))
    
def clear2():
    rolent.delete(first=0,last=len(rolent.get()))
    
def clear3():
    nament.delete(first=0,last=len(nament.get()))
    
def track():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath) 
    df=pd.read_csv("StudentDetails\studentDetails.csv")
    cam=cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray,1.3,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            print(conf)                                   
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                
            else:
                Id='Unknown'                
                tt=str(Id)  
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    #print(attendance)
    #res=attendance
    #msg.showinfo("info",res)
def register():
    global rolent,nament,home
    home=t.Tk()
    home.title("REGISTRATION")
    home.geometry('800x700')
    roll=t.Label(home,text="ROLL NUMBER")
    name=t.Label(home,text="NAME")
    rolent=t.Entry(home,width=20)
    nament=t.Entry(home,width=20)
    clr2=t.Button(home,text="CLEAR",command=clear2)
    clr3=t.Button(home,text="CLEAR",command=clear3)
    #enr=t.Button(home,text="ENROLL FACE",command=enroll)
    #enr.place(x=20,y=200)    
    ver=t.Button(home,text="VERIFY",command=verify)    
    tra=t.Button(home,text="TRAIN FACES",command=train)
    #sub=t.Button(home,text="SUBMIT",command=submit)
    roll.place(x=10,y=50)
    name.place(x=10,y=80)
    rolent.place(x=100,y=50)
    nament.place(x=100,y=80)
    clr2.place(x=250,y=50)
    clr3.place(x=250,y=80)
    ver.place(x=100,y=120)
    
    tra.place(x=120,y=200)
    #sub.place(x=220,y=200)
def attendance():
    home=t.Tk()
    home.title("ATTENDANCE")
    home.geometry('800x700')
    tra=t.Button(home,text="TRACK ATTENDANCE",command=track)
    tra.place(x=100,y=600)   

def homescreen():
    home=t.Tk()
    home.title("HOME SCREEN")
    home.geometry('800x700')
    reg=t.Button(home,text="REGISTER",command=register)
    att=t.Button(home,text="ATTENDANCE",command=attendance)
    reg.place(x=120,y=75)
    att.place(x=200,y=75)

def login1():
    global userent,pasent
    a='pj'
    p='123'
    if userent.get()==a:
        if pasent.get()==p:
            #print('yes')
           
            login.destroy()
            homescreen()
        else:
            msg.showwarning("warning","Wrong Password")
            clear1()
    else:
        msg.showwarning("warning","Invalid Credintals")
        clear()
        clear1() 
        

login=t.Tk()
login.title("LOGIN")
login.geometry('400x150')
user=t.Label(login,text="ENTER USERNAME")
pas=t.Label(login,text="ENTER PASSWORD")
clr=t.Button(login,text="CLEAR",command=clear1).place(x=250,y=80)
clr1=t.Button(login,text="CLEAR",command=clear).place(x=250,y=50)
userent=t.Entry(login,width=20)
pasent=t.Entry(login,width=20,show='*')
sub=t.Button(login,text="SUBMIT",command=login1)
user.place(x=10,y=50)
pas.place(x=10,y=80)
userent.place(x=120,y=50)
pasent.place(x=120,y=80)
sub.place(x=50,y=120)
login.mainloop()