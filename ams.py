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
clgname="SAI TIRUMALA NVR ENGINEERING COLLEGE"
def verify():
    global r,n
    r=rolent.get()
    n=nament.get()    
    if len(r)==10 and len(n)>=1:
        msg.showinfo("info","Verification Successfull")
        enr=t.Button(home,text="ENROLL FACE",command=enroll,bg='snow',height=2)
        enr.place(x=230,y=320)
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
            if(conf < 75):
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
        if (cv2.waitKey(20)==ord('q')):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    print(attendance)
    att=attendance
    display(att)
    
def display(att):
    home=t.Tk()
    home.title("ATTENDANCE")
    home.geometry('500x250')
    home.configure(bg='springgreen')
    lab=t.Label(home,text=att,width=50,height=2,bg='snow')
    lab.place(x=50,y=50)
    
def register():
    global rolent,nament,home
    home=t.Tk()
    home.title("REGISTRATION")
    home.geometry('700x500')
    home.configure(bg='spring green')
    roll=t.Label(home,text="ROLL NUMBER",width=15,height=1,bg='snow',font='Arial 12')
    name=t.Label(home,text="NAME",width=15,height=1,bg='snow',font='Arial 12')
    rolent=t.Entry(home,width=20,font='Arial 16')
    nament=t.Entry(home,width=20,font='Arial 16')
    clr2=t.Button(home,text="CLEAR",command=clear2,width=10,height=2,bg='snow')
    clr3=t.Button(home,text="CLEAR",command=clear3,width=10,height=2,bg='snow') 
    ver=t.Button(home,text="VERIFY",command=verify,width=10,height=2,bg='snow')    
    tra=t.Button(home,text="TRAIN FACES",command=train,width=10,height=2,bg='snow')
    roll.place(x=70,y=150)
    name.place(x=70,y=200)
    rolent.place(x=230,y=150)
    nament.place(x=230,y=200)
    clr2.place(x=510,y=150)
    clr3.place(x=510,y=200)
    ver.place(x=100,y=320)
    tra.place(x=350,y=320)
    
def attendance():
    home=t.Tk()
    home.title("ATTENDANCE")
    home.geometry('500x250')
    home.configure(bg='turquoise1')
    tra=t.Button(home,text="TRACK ATTENDANCE",command=track)
    tra.place(x=180,y=140)  
    
def homescreen():
    home=t.Tk()
    home.title("HOME SCREEN")
    home.geometry('500x250')
    home.configure(bg='cyan')
    reg=t.Button(home,text="REGISTER",command=register)
    att=t.Button(home,text="ATTENDANCE",command=attendance)
    reg.place(x=140,y=85)
    att.place(x=230,y=85)

def login1():
    global userent,pasent
    a='pj'
    p='123'
    if userent.get()==a:
        if pasent.get()==p:
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
login.geometry('500x300')
login.configure(bg='lightslateblue')
user=t.Label(login,text="USERNAME",height=1,width=17)
pas=t.Label(login,text="PASSWORD",height=1,width=17)
clg=t.Label(login,text=clgname,height=2,width=50,font='BOLD',bg='lightslateblue')
clr=t.Button(login,text="CLEAR",height=1,width=10,command=clear1)
clr.place(x=350,y=120)
clr1=t.Button(login,text="CLEAR",height=1,width=10,command=clear)
clr1.place(x=350,y=70)
userent=t.Entry(login,width=15,font='Roman')
pasent=t.Entry(login,width=15,show='*',font='Roman')
sub=t.Button(login,text="SUBMIT",command=login1,width=10,height=1)
user.place(x=15,y=70)
pas.place(x=15,y=120)
userent.place(x=180,y=70)
pasent.place(x=180,y=120)
sub.place(x=210,y=180)
clg.place(x=0,y=250)
login.mainloop()