import cv2
n_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
l_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')
def cuttz(bwimage,cimage):
    face=n_cascade.detectMultiScale(bwimage,1.3,5)
    for(x,y,w,h) in face:
        cv2.rectangle(cimage,(x,y),(x+w,y+h),(0,255,0),3)
        r_bw=bwimage[y:y+h,x:x+w]
        s_cl=cimage[y:y+h,x:x+h]
        eye=l_cascade.detectMultiScale(r_bw,1.5,10)
        for(ex,ey,ew,eh) in eye:
            cv2.rectangle(s_cl,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
    return cimage
vcap=cv2.VideoCapture(0)
while True:
    _,cimage=vcap.read()
    bwimage=cv2.cvtColor(cimage,cv2.COLOR_BGR2GRAY)

    my=cuttz(bwimage,cimage)
    cv2.imshow('Video',my)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
vcap.release()
cv2.destroyAllWindows()           


