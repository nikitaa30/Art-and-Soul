Art & Soul
===
**Art and Soul is a Website based on Neural-Style-Transfer.**
---

It is built on **Django**.
It consists of the following Utilities-
1. Login and Signup.

2. Select a File that you wish to **upload** -**OR**-

   **Draw a Doodle** by clicking on the button. It will open a Paint-like Window for you to draw.
   
3. Select the **Style** you want your Image to be painted in.

4. Click on submit and 
**LET THE MAGIC HAPPEN!**

![alt text](https://raw.githubusercontent.com/nikitaa30/Art-and-Soul/master/media/None/ss1.png)



![alt text](https://raw.githubusercontent.com/nikitaa30/Art-and-Soul/master/media/None/ss2.png)

Setup Details
---
1. Django==2.2.0
2. keras==2.2.4
3. matplotlib==3.0.3
4. Tensorflow==1.8.0(CPU OR GPU version)
5. Tkinter

Neural-Network Architecture
---
The model used is **VGG16** for the purpose of Style-Transfer.
Reason being, It is relatively simpler, computationally less expensive and provides decent results.

Draw A Doodle with Tkinter
---
This utility package allows you to create Canvas and edit them. I have built a Custom class using basic **Event handling** that allows you to draw on a canvas and
-Paint with a Brush
-Draw with a Pencil
-Use Eraser
-Increase and decrease Brush Stroke Width
-Choose Colors
-Save your Doodle

![alt text](https://raw.githubusercontent.com/nikitaa30/Art-and-Soul/master/media/None/Capture.PNG)

Website Details
---
The Website is based solely on **Django**.
The Templates are made with
-html
-css
-javascript
-bootstrap
**The Link of the Website Would be added here SOON!**

Results
---
You can choose and Add Various styles as per your choice and the output would change accordingly.
---
**Input Files**
![alt text](https://raw.githubusercontent.com/nikitaa30/Art-and-Soul/master/media/None/out.PNG)


**Style-Transfered File**


![alt text](https://raw.githubusercontent.com/nikitaa30/Art-and-Soul/master/media/None/out2.PNG)


**Transition over the iterations**


![alt text](https://raw.githubusercontent.com/nikitaa30/Art-and-Soul/master/media/None/xyz.gif)



