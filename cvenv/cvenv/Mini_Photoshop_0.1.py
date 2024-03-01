from tkinter import * 
from tkinter import filedialog as fd
import cv2 as cv
from PIL import Image, ImageTk
import imutils
import random, numpy as np
from matplotlib import pyplot as plt

# Root window
root = Tk()
root.title('Mini')
root.geometry('800x600')
canvas = Canvas()
panelA = None
Brightness =DoubleVar()
Contrast = DoubleVar()
R=DoubleVar()
G=DoubleVar()
B=DoubleVar()
H=DoubleVar()
S=DoubleVar()
V=DoubleVar()
# Initial Value
R_Value=0.0
G_Value=0.0
B_Value=0.0
H_Value=0.0
S_Value=0.0
V_Value=0.0
alpha_b=1.0
gamma_b=0.0
alpha_c=1.0
gamma_c=0.0
dim = (0,0)
gray_flag = 0
invert_flag = 0
Hist_flag = 0 
Salt_Pepper_flag = 0
Gaussian_flag = 0
filter1_state = 0
filter2_state = 0
filter3_state = 0
filter4_state = 0
filter5_state = 0
stateVar1=IntVar()
stateVar2=IntVar()
stateVar3=IntVar()
stateVar4=IntVar()
stateVar5=IntVar()
add_enable = 0
Added_Image = None
Blend_enable = 0
Blend_Image = None
Opacity_Value1 = DoubleVar()
Opacity_Value2 = DoubleVar()
Apply_Opacity_Value1 = 100
Apply_Opacity_Value2 = 100
sub_enable = 0
sub_Image = None
masking_enable = 0
full_buf_img = None
masking_zone =(0,0,0,0)
masking_img_trigger = 0
stored_flag=0
whitebalance_enable=0
Hist_garph = 0 
# Text editor
def Update_Image(Input_Image):
    global buf,Added_Image,sub_Image,masking_img,masking_img_trigger,Blend_Image,black,white,probs
    if masking_enable == 0 and masking_img_trigger == 1 :
        Input_Image[int(masking_zone[1]):int(masking_zone[1]+masking_zone[3]),int(masking_zone[0]):int(masking_zone[0]+masking_zone[2])] = masking_img
        print("Masking")
    Input_Image =cv. cvtColor(Input_Image, cv.COLOR_BGR2RGB)
    buf = cv.addWeighted(Input_Image, alpha_b, Input_Image, 0, gamma_b)
    buf = cv.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    RChannel=cv.addWeighted(buf[:,:,0],1,buf[:,:,0],0,R_Value)
    GChannel=cv.addWeighted(buf[:,:,1],1,buf[:,:,1],0,G_Value)
    BChannel=cv.addWeighted(buf[:,:,2],1,buf[:,:,2],0,B_Value)
    buf[:,:,0]=RChannel
    buf[:,:,1]=GChannel
    buf[:,:,2]=BChannel
    buf = cv.cvtColor(buf, cv.COLOR_RGB2HSV)
    HChannel=cv.addWeighted(buf[:,:,0],1,buf[:,:,0],0,H_Value)
    SChannel=cv.addWeighted(buf[:,:,1],1,buf[:,:,1],0,S_Value)
    VChannel=cv.addWeighted(buf[:,:,2],1,buf[:,:,2],0,V_Value)
    buf[:,:,0]=HChannel
    buf[:,:,1]=SChannel
    buf[:,:,2]=VChannel
    buf = cv.cvtColor(buf, cv.COLOR_HSV2BGR)
    print("g",gray_flag)
    print("s",Salt_Pepper_flag)
    print("H",Hist_flag)
    if add_enable == 1 :
        Added_Image = cv.resize(Added_Image, ( buf.shape[1], buf.shape[0] ), interpolation = cv.INTER_AREA)
        buf = cv.addWeighted(buf,1,Added_Image,1,0)
        
    if sub_enable == 1  :
        sub_Image = cv.resize(sub_Image, ( buf.shape[1], buf.shape[0] ), interpolation = cv.INTER_AREA)
        buf = cv.subtract(buf,sub_Image)
    
    if Blend_enable == 1  :
        Blend_Image = cv.resize(Blend_Image, ( buf.shape[1], buf.shape[0] ), interpolation = cv.INTER_AREA)
        print("Apply_Opacity_Value1 =",Apply_Opacity_Value1)
        print("Apply_Opacity_Value2 =",Apply_Opacity_Value2)
        buf = cv.addWeighted(buf,Apply_Opacity_Value1/100.0,Blend_Image,Apply_Opacity_Value2/100.0,0)
    
    if gray_flag == 1:
        buf = cv.cvtColor(buf, cv.COLOR_BGR2GRAY)
    if Salt_Pepper_flag == 1 :
            prob = 0.5
            buf[probs < (prob / 2)] = black
            buf[probs > 1 - (prob / 2)] = white
    if Gaussian_flag == 1 :
        output = np.array(buf / 255, dtype=float)
        print("Output =",output)
        noise = np.random.normal(0, 0.1 ** 0.5, output.shape)
        out_g = output + noise
        if out_g.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out_g, low_clip, 1.0)
        out_g = np.uint8(out_g * 255)
        buf = out_g
    if invert_flag == 1:
        buf = cv.bitwise_not(buf)
    if Hist_flag == 1 and gray_flag ==1:    
        buf = cv.equalizeHist(buf)
    if Hist_flag == 1 and gray_flag !=1 :
        buf = cv.cvtColor(buf, cv.COLOR_BGR2YCrCb)
        # equalize the histogram of the Y channel
        buf[:, :, 0] = cv.equalizeHist(buf[:, :, 0])
            # convert back to RGB color-space from YCrCb
        buf = cv.cvtColor(buf, cv.COLOR_YCrCb2BGR)
    if filter1_state == 1 :
        buf = cv.blur(buf, (3,3))
    if filter2_state == 1 :
        buf = cv.GaussianBlur(buf, (3,3), 0)
    if filter3_state == 1 :
        buf = cv.medianBlur(buf, 3)
    if filter4_state == 1 :
        kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
        buf = cv.filter2D(src=buf, ddepth=-1, kernel=kernel)
    if filter5_state == 1 :
        buf = cv.Canny(buf, 125, 175)
    if masking_enable == 1:
        if len(buf.shape)<3:
            print("Do")
            buf=cv.cvtColor(buf,cv.COLOR_GRAY2RGB)
        masking_img = buf
        full_buf_img[int(masking_zone[1]):int(masking_zone[1]+masking_zone[3]),int(masking_zone[0]):int(masking_zone[0]+masking_zone[2])] = buf
        buf = full_buf_img
        masking_img_trigger =1
    
    if whitebalance_enable == 1:
        if len(buf.shape)==2:
            print("Do")
            buf=cv.cvtColor(buf,cv.COLOR_GRAY2BGR)
        b, g, r = cv.split(buf)
        r_avg = cv.mean(r)[0]
        g_avg = cv.mean(g)[0]
        b_avg = cv.mean(b)[0]
        if r_avg != 0 and g_avg != 0 and b_avg != 0:
            k = (r_avg + g_avg + b_avg)/3
            kr = k/r_avg
            kg = k/g_avg
            kb = k/b_avg
            r = cv.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
            g = cv.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
            b = cv.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
            buf = cv.merge([b, g, r])
        else:
            print("Cannot Do White balance")
        
    if gray_flag == 0:
        display_img = cv.cvtColor(buf, cv.COLOR_BGR2RGB)
    else:
        display_img = buf
    img_display = Image.fromarray(display_img)
    img_display = ImageTk.PhotoImage(img_display)
    panelA.configure(image=img_display)
    panelA.image = img_display
    if Hist_garph == 1 :
        plt.close()
        if len(display_img.shape)==2:
            histr = cv.calcHist([display_img],[0],None,[256],[0,256])
            plt.plot(histr)
            plt.xlim([0,256])
        else:
            color = ('r','g','b')
            for i,col in enumerate(color):
                histr = cv.calcHist([display_img],[i],None,[256],[0,256])
                plt.plot(histr,color = col)
                plt.xlim([0,256])
        plt.show()
    else:
        plt.close()
    Label(Image_Frame,text = "Height:" + str(dim[0]) +"px "+"Width:"+ str(dim[1]) + "px" ).grid(row=1, column=0)
    print("All Filter",filter1_state,filter2_state,filter3_state,filter4_state,filter5_state)


def RGB_Scale():
    HSV_Tool_Frame.place_forget()
    BC_Tool_Frame.place_forget()
    Resize_Tool_Frame.place_forget()
    Rotate_Tool_Frame.place_forget()
    Noise_Tool_Frame.place_forget()
    Filter_Tool_Frame.place_forget()
    Blend_Tool_Frame.place_forget()
    RGB_Tool_Frame.place(relx=0.95,rely=0.5,anchor=E)
    R_Label=Label(RGB_Tool_Frame, text='R').grid(row=0,column=0)
    R_scale = Scale( RGB_Tool_Frame,orient=HORIZONTAL, variable= R, command=RGB_Receive,from_=-255,to=255).grid(row=1,column=0)
    G_Label=Label(RGB_Tool_Frame, text='G').grid(row=2,column=0)
    G_scale = Scale( RGB_Tool_Frame ,orient=HORIZONTAL, variable = G, command=RGB_Receive,from_=-255,to=255).grid(row=3,column=0)
    B_Label=Label(RGB_Tool_Frame, text='B').grid(row=4,column=0)
    B_scale = Scale( RGB_Tool_Frame,orient=HORIZONTAL, variable= B, command=RGB_Receive,from_=-255,to=255).grid(row=5,column=0)

def HSV_Scale():
    RGB_Tool_Frame.place_forget()
    BC_Tool_Frame.place_forget()
    Resize_Tool_Frame.place_forget()
    Rotate_Tool_Frame.place_forget()
    Noise_Tool_Frame.place_forget()
    Filter_Tool_Frame.place_forget()
    Blend_Tool_Frame.place_forget()
    HSV_Tool_Frame.place(relx=0.95,rely=0.5,anchor=E)
    H_Label=Label(HSV_Tool_Frame, text='H').grid(row=0,column=0)
    H_scale = Scale( HSV_Tool_Frame,orient=HORIZONTAL, variable= H, command=HSV_Receive,from_=-255,to=255).grid(row=1,column=0)
    S_Label=Label(HSV_Tool_Frame, text='S').grid(row=2,column=0)
    S_scale = Scale( HSV_Tool_Frame ,orient=HORIZONTAL, variable = S, command=HSV_Receive,from_=-255,to=255).grid(row=3,column=0)
    V_Label=Label(HSV_Tool_Frame, text='V').grid(row=4,column=0)
    V_scale = Scale( HSV_Tool_Frame,orient=HORIZONTAL, variable= V, command=HSV_Receive,from_=-255,to=255).grid(row=5,column=0)

def scale_generate():
    HSV_Tool_Frame.place_forget()
    RGB_Tool_Frame.place_forget()
    Resize_Tool_Frame.place_forget()
    Rotate_Tool_Frame.place_forget()
    Noise_Tool_Frame.place_forget()
    Filter_Tool_Frame.place_forget()
    Blend_Tool_Frame.place_forget()
    BC_Tool_Frame.place(relx=0.95,rely=0.5,anchor=E)
    Brightness_Label=Label(BC_Tool_Frame, text='Brightness').grid(row=0,column=0)
    Brightness_scale = Scale( BC_Tool_Frame,orient=HORIZONTAL, variable= Brightness, command=funcBrightContrast,from_=-255,to=255).grid(row=1,column=0)
    Contrast_Label=Label(BC_Tool_Frame, text='Contrast').grid(row=2,column=0)
    Contrast_scale = Scale( BC_Tool_Frame ,orient=HORIZONTAL, variable = Contrast, command=funcBrightContrast,from_=-127,to=127).grid(row=3,column=0)
    
def Crop_function():
    global current_img
    Crop_Image = current_img
    Crop_Zone = cv.selectROI("Crop(C=Cancel Enter=Crop)", Crop_Image)
    print(Crop_Zone)
    if Crop_Zone != (0,0,0,0):
        Crop_Image = Crop_Image[int(Crop_Zone[1]):int(Crop_Zone[1]+Crop_Zone[3]), 
                        int(Crop_Zone[0]):int(Crop_Zone[0]+Crop_Zone[2])]
        current_img = Crop_Image
        Update_Image(current_img)
    
def Undo_Crop():
    global current_img,original_img
    current_img = original_img
    Update_Image(current_img)
    
def Get_Hight_Width():
    global dim,current_img,masking_enable
    if masking_enable == 0:
        try:
            hight_value=int(e1.get())
            width_value=int(e2.get())
        except:
            print("Wrong Value")
            
        try:
            dim =(width_value, hight_value)
            print(dim)
            current_img =cv.resize(current_img, dim, interpolation = cv.INTER_AREA)
            Update_Image(current_img)
        except:
            pass
    
def Flip_H_function():
    global 	current_img
    flipHorizontal = cv2.flip(current_img, 1)
    current_img = flipHorizontal
    Update_Image(current_img)

def Flip_H_function():
    global 	current_img
    flipHorizontal = cv.flip(current_img, 1)
    current_img = flipHorizontal
    Update_Image(current_img)    

def Flip_V_function():
    global 	current_img
    flipVertical = cv.flip(current_img, 0)
    current_img = flipVertical
    Update_Image(current_img)    
    
def Resize_Menu():
    global e1,e2,masking_enable
    HSV_Tool_Frame.place_forget()
    RGB_Tool_Frame.place_forget()
    BC_Tool_Frame.place_forget()
    Noise_Tool_Frame.place_forget()
    Rotate_Tool_Frame.place_forget()
    Filter_Tool_Frame.place_forget()
    Blend_Tool_Frame.place_forget()
    Resize_Tool_Frame.place(relx=0.95,rely=0.5,anchor=E)
    Label(Resize_Tool_Frame, 
         text="Height").grid(row=0)
    Label(Resize_Tool_Frame, 
         text="Width").grid(row=1)
    e1 = Entry(Resize_Tool_Frame)
    e2 = Entry(Resize_Tool_Frame)
    e1.grid(row=0, column=1)
    e2.grid(row=1, column=1)
    if masking_enable == 0:
        Button(Resize_Tool_Frame,text="Apply",command=Get_Hight_Width, height=1, width=10).grid(row=2,column=0)

def Rotate_By_Degree():
    global current_img,original_img
    Resize_Before_Rotate = cv.resize(original_img, dim, interpolation = cv.INTER_AREA)
    current_img = imutils.rotate_bound(Resize_Before_Rotate, int(e3.get()))
    Update_Image(current_img)

def Rotate_CW():
    global current_img
    Resize_Before_Rotate = cv.resize(current_img, dim, interpolation = cv.INTER_AREA)
    current_img = cv.rotate(Resize_Before_Rotate, cv.ROTATE_90_CLOCKWISE)
    Update_Image(current_img)

def Rotate_CCW():
    global current_img
    Resize_Before_Rotate = cv.resize(current_img, dim, interpolation = cv.INTER_AREA)
    current_img = cv.rotate(Resize_Before_Rotate, cv.ROTATE_90_COUNTERCLOCKWISE)
    Update_Image(current_img)
   
def Rotate_function():
    global e3
    HSV_Tool_Frame.place_forget()
    RGB_Tool_Frame.place_forget()
    Resize_Tool_Frame.place_forget()
    Noise_Tool_Frame.place_forget()
    Filter_Tool_Frame.place_forget()
    Blend_Tool_Frame.place_forget()
    Rotate_Tool_Frame.place(relx=0.95,rely=0.5,anchor=E)
    Button(Rotate_Tool_Frame,text="CW",command=Rotate_CW,height=1, width=5).grid(row=0,column=0)
    Button(Rotate_Tool_Frame,text="CCW",command=Rotate_CCW, height=1, width=5).grid(row=0,column=3)
    Label(Rotate_Tool_Frame, 
         text="Degree").grid(row=1,column=0)
    e3 = Entry(Rotate_Tool_Frame, width=10)
    e3.grid(row=1, column=1)
    Button(Rotate_Tool_Frame,text="Apply",command=Rotate_By_Degree, height=1, width=10).grid(row=2,column=1)
    
def Convert_To_Gray():
    global current_img,gray_flag
    gray_flag = 1 - gray_flag
    Update_Image(current_img)

def Convert_To_Invert():
    global current_img,invert_flag
    invert_flag = 1 - invert_flag
    Update_Image(current_img)
    
def Hist_toggle():
    global current_img,Hist_flag
    Hist_flag = 1 - Hist_flag
    Update_Image(current_img)

def Salt_Pepper():
    global current_img,Salt_Pepper_flag,buf,black,white,probs
    Salt_Pepper_flag = 1 - Salt_Pepper_flag
    output = current_img.copy()
    if len(current_img.shape) == 2:
        black = 0
        white = 255
    else:
        colorspace = current_img.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    Update_Image(current_img)
    
def Gaussian():
    global current_img,Gaussian_flag
    Gaussian_flag = 1 - Gaussian_flag
    Update_Image(current_img)
    
def Noise_Frame():
    HSV_Tool_Frame.place_forget()
    RGB_Tool_Frame.place_forget()
    Resize_Tool_Frame.place_forget()
    Filter_Tool_Frame.place_forget()
    Blend_Tool_Frame.place_forget()
    Noise_Tool_Frame.place(relx=0.95,rely=0.5,anchor=E)
    Label(Noise_Tool_Frame,text="salt-and-pepper").grid(row=0,column=0)
    Button(Noise_Tool_Frame,command=Salt_Pepper, height=1, width=10).grid(row=1,column=0)
    Label(Noise_Tool_Frame,text="Gaussian").grid(row=2,column=0)
    Button(Noise_Tool_Frame,command=Gaussian, height=1, width=10).grid(row=3,column=0)

def filter1():
    global current_img,filter1_state
    filter1_state = 1-filter1_state
    Update_Image(current_img)

def filter2():
    global current_img,filter2_state
    filter2_state = 1-filter2_state
    Update_Image(current_img)
    
def filter3():
    global current_img,filter3_state
    filter3_state = 1-filter3_state
    Update_Image(current_img)
    
def filter4():
    global current_img,filter4_state
    filter4_state = 1-filter4_state
    Update_Image(current_img)

def filter5():
    global current_img,filter5_state
    filter5_state = 1-filter5_state
    Update_Image(current_img)    
    
def Filter_Frame():
    HSV_Tool_Frame.place_forget()
    RGB_Tool_Frame.place_forget()
    Resize_Tool_Frame.place_forget()
    Noise_Tool_Frame.place_forget()
    Blend_Tool_Frame.place_forget()
    Filter_Tool_Frame.place(relx=0.95,rely=0.5,anchor=E)
    Button(Filter_Tool_Frame, text='Mean',command=filter1,height=1, width=10).grid(row=0,column=0)
    Button(Filter_Tool_Frame, text='Gaussian',command=filter2,height=1, width=10).grid(row=1,column=0)
    Button(Filter_Tool_Frame, text='Median',command=filter3, height=1, width=10).grid(row=2,column=0)
    Button(Filter_Tool_Frame, text='Sharpen ',command=filter4, height=1, width=10).grid(row=3,column=0)
    Button(Filter_Tool_Frame, text='edge  ',command=filter5, height=1, width=10).grid(row=4,column=0)

def Add_Func():
    global current_img,add_enable,Added_Image
    add_enable = 1 - add_enable
    if add_enable ==1 :
        filetypes = (
            ('JPG', '*.jpg'),
            ('All files', '*.*')
        )
        # show the open file dialog
        path = fd.askopenfilename(filetype=filetypes)
        if len(path) != 0  :
            Added_Image = cv.imread(path)
    Update_Image(current_img)
    
def Subtract_Func():
    global current_img,sub_enable,sub_Image
    sub_enable = 1 - sub_enable
    if sub_enable ==1 :
        filetypes = (
            ('JPG', '*.jpg'),
            ('All files', '*.*')
        )
        # show the open file dialog
        path = fd.askopenfilename(filetype=filetypes)
        if len(path) != 0  :
            sub_Image = cv.imread(path)
    Update_Image(current_img)  

def Masking_Function():
    global masking_zone,current_img,masking_enable,buf,full_buf_img,original_img,masking_image
    global old_add_enable,add_enable,old_sub_enable,sub_enable,old_Blend_enable,Blend_enable,old_gray_flag,gray_flag
    global old_Salt_Pepper_flag,Salt_Pepper_flag,old_Gaussian_flag,Gaussian_flag,old_invert_flag,invert_flag 
    global old_Hist_flag,Hist_flag,old_filter1_state,filter1_state,old_filter2_state,filter2_state,old_filter3_state,filter3_state
    global old_filter4_state,filter4_state,old_filter5_state,filter5_state,old_porbs,porbs,old_Blend_Image,Blend_Image,old_sub_Image,sub_Image,old_Added_Image,Added_Image
    global old_R_Value,R_Value,old_G_Value,G_Value,old_B_Value,B_Value,old_H_Value,H_Value,old_S_Value,S_Value,old_V_Value,V_Value
    global old_alpha_b,alpha_b,old_gamma_b,gamma_b,old_alpha_c,alpha_c,old_gamma_c,gamma_c,stored_flag,Apply_Opacity_Value1,old_Apply_Opacity_Value1
    global Apply_Opacity_Value2,old_Apply_Opacity_Value2
    Resize_Tool_Frame.place_forget()
    masking_enable = 1-masking_enable
    if masking_enable == 1:
        masking_zone =cv.selectROI("Select(C=Cancel Enter=Apply)", buf)
        print(masking_zone)
        if masking_zone != (0,0,0,0):
            Button(Tool_Frame,text="Unmask",command=Masking_Function, height=1, width=10).grid(row=8,column=1)
            full_buf_img = buf
            current_img = current_img[int(masking_zone[1]):int(masking_zone[1]+masking_zone[3]), 
                        int(masking_zone[0]):int(masking_zone[0]+masking_zone[2])]
            old_add_enable = add_enable
            old_sub_enable = sub_enable
            old_Blend_enable = Blend_enable
            old_gray_flag = gray_flag
            old_Salt_Pepper_flag = Salt_Pepper_flag
            old_Gaussian_flag = Gaussian_flag
            old_invert_flag = invert_flag 
            old_Hist_flag = Hist_flag
            old_filter1_state = filter1_state
            old_filter2_state = filter2_state
            old_filter3_state = filter3_state
            old_filter4_state = filter4_state
            old_filter5_state = filter5_state
            if Salt_Pepper_flag == 1:
                old_porbs = porbs
            if Blend_enable ==1:
                old_Blend_Image = Blend_Image
            if sub_enable == 1:
                old_sub_Image = sub_Image
            if add_enable == 1:
                old_Added_Image = Added_Image
            old_Apply_Opacity_Value1 = Apply_Opacity_Value1
            old_Apply_Opacity_Value2 = Apply_Opacity_Value2
            old_R_Value=R_Value
            old_G_Value=G_Value
            old_B_Value=B_Value
            old_H_Value=H_Value
            old_S_Value=S_Value
            old_V_Value=V_Value
            old_alpha_b=alpha_b
            old_gamma_b=gamma_b
            old_alpha_c=alpha_c
            old_gamma_c=gamma_c
            stored_flag = 1
            Update_Image(current_img)
        else:
            masking_enable = 0
            
    if masking_enable == 0:
        if stored_flag ==1 :
            Button(Tool_Frame,text="Masking",command=Masking_Function, height=1, width=10).grid(row=8,column=1)
            stored_flag = 0
            current_img =cv.resize(original_img, dim, interpolation = cv.INTER_AREA)
            add_enable = old_add_enable
            sub_enable = old_sub_enable
            Blend_enable = old_Blend_enable
            gray_flag = old_gray_flag
            Salt_Pepper_flag = old_Salt_Pepper_flag
            Gaussian_flag = old_Gaussian_flag
            invert_flag = old_invert_flag 
            Hist_flag = old_Hist_flag
            filter1_state = old_filter1_state
            filter2_state = old_filter2_state
            filter3_state = old_filter3_state
            filter4_state = old_filter4_state
            filter5_state = old_filter5_state
            if Salt_Pepper_flag == 1:
                    porbs = old_porbs
            if Blend_enable ==1:
                Blend_Image = old_Blend_Image
            if sub_enable == 1:
                sub_Image = old_sub_Image
            if add_enable == 1:
                Added_Image = old_Added_Image
            Apply_Opacity_Value1 = old_Apply_Opacity_Value1
            Apply_Opacity_Value2 = old_Apply_Opacity_Value2
            R_Value=old_R_Value
            G_Value=old_G_Value
            B_Value=old_B_Value
            H_Value=old_H_Value
            S_Value=old_S_Value
            V_Value=old_V_Value
            alpha_b=old_alpha_b
            gamma_b=old_gamma_b
            alpha_c=old_alpha_c
            gamma_c=old_gamma_c
        Update_Image(current_img)

def Reset_Image():
    global current_img,original_img,masking_img_trigger,masking_enable
    if masking_enable == 0 and masking_img_trigger == 1 :
        masking_img_trigger = 0
        masking_enable = 0
        current_img =cv.resize(original_img, dim, interpolation = cv.INTER_AREA)
        print("masking_img_trigger =",masking_img_trigger)
        Update_Image(current_img)

def get_opacity(x):
    global Apply_Opacity_Value1,Apply_Opacity_Value2
    Apply_Opacity_Value1 = Opacity_Value1.get()
    Apply_Opacity_Value2 = Opacity_Value2.get()
    Update_Image(current_img)
    
def Blend_Function():
    global current_img,Blend_enable,Blend_Image
    Blend_enable = 1 - Blend_enable
    if Blend_enable ==1 :
        filetypes = (
            ('JPG', '*.jpg'),
            ('All files', '*.*')
        )
        # show the open file dialog
        path = fd.askopenfilename(filetype=filetypes)
        if len(path) != 0  :
            Blend_Image = cv.imread(path)
            HSV_Tool_Frame.place_forget()
            RGB_Tool_Frame.place_forget()
            Resize_Tool_Frame.place_forget()
            Noise_Tool_Frame.place_forget()
            Filter_Tool_Frame.place_forget()
            Blend_Tool_Frame.place(relx=0.95,rely=0.5,anchor=E)
            Label(Blend_Tool_Frame, text='Img1 Opacity').grid(row=0,column=0)
            Scale( Blend_Tool_Frame,orient=HORIZONTAL, variable=Opacity_Value1,command=get_opacity,from_=0,to=100).grid(row=1,column=0)
            Label(Blend_Tool_Frame, text='Img2 Opacity').grid(row=2,column=0)
            Scale( Blend_Tool_Frame,orient=HORIZONTAL, variable=Opacity_Value2,command=get_opacity,from_=0,to=100).grid(row=3,column=0)
            for item in Blend_Tool_Frame.winfo_children():
                if type(item) == type(Scale()):
                    item.set(100)
    if Blend_enable ==0:
        Blend_Tool_Frame.place_forget()
    Update_Image(current_img)

def White_Balance_Button():
    global whitebalance_enable
    whitebalance_enable = 1 - whitebalance_enable
    Update_Image(current_img)
    
def RGBGraph():
    global Hist_garph
    Hist_garph = 1 - Hist_garph
    Update_Image(current_img)
    
def Generate_Button():
    global HSV_Tool_Frame,BC_Tool_Frame,RGB_Tool_Frame,Resize_Tool_Frame,Rotate_Tool_Frame,Noise_Tool_Frame,Filter_Tool_Frame,Blend_Tool_Frame
    Button(Tool_Frame,text="RGB",command=RGB_Scale, height=1, width=10).grid(row=0,column=0)
    Button(Tool_Frame,text="HSV",command=HSV_Scale, height=1, width=10).grid(row=1,column=0)
    Button(Tool_Frame,text="BC",command=scale_generate, height=1, width=10).grid(row=2,column=0)
    Button(Tool_Frame,text="Crop",command=Crop_function, height=1, width=10).grid(row=3,column=0)
    Button(Tool_Frame,text="Undo Crop",command=Undo_Crop, height=1, width=10).grid(row=4,column=0)
    Button(Tool_Frame,text="Resize",command=Resize_Menu, height=1, width=10).grid(row=5,column=0)
    Button(Tool_Frame,text="FlipH",command=Flip_H_function, height=1, width=10).grid(row=6,column=0)
    Button(Tool_Frame,text="FlipV",command=Flip_V_function, height=1, width=10).grid(row=0,column=1)
    Button(Tool_Frame,text="Rotate",command=Rotate_function, height=1, width=10).grid(row=1,column=1)
    Button(Tool_Frame,text="Gray",command=Convert_To_Gray, height=1, width=10).grid(row=2,column=1)
    Button(Tool_Frame,text="Invert",command=Convert_To_Invert, height=1, width=10).grid(row=3,column=1)
    Button(Tool_Frame,text="Hist_E",command=Hist_toggle, height=1, width=10).grid(row=4,column=1)
    Button(Tool_Frame,text="Noise",command=Noise_Frame, height=1, width=10).grid(row=5,column=1)
    Button(Tool_Frame,text="Filter",command=Filter_Frame, height=1, width=10).grid(row=6,column=1)
    Button(Tool_Frame,text="Add",command=Add_Func, height=1, width=10).grid(row=7,column=0)
    Button(Tool_Frame,text="Subtract",command=Subtract_Func, height=1, width=10).grid(row=7,column=1)
    Button(Tool_Frame,text="Blend",command=Blend_Function, height=1, width=10).grid(row=8,column=0)
    Button(Tool_Frame,text="Masking",command=Masking_Function, height=1, width=10).grid(row=8,column=1)
    Button(Tool_Frame,text="Reset Mask",command=Reset_Image, height=1, width=10).grid(row=9,column=0)
    Button(Tool_Frame,text="White Bl",command=White_Balance_Button, height=1, width=10).grid(row=9,column=1)
    Button(Tool_Frame,text="RGBGraph",command=RGBGraph, height=1, width=10).grid(row=10,column=0)
    RGB_Tool_Frame = LabelFrame(root,text="RGB Scale")
    BC_Tool_Frame = LabelFrame(root,text="BC Scale")
    HSV_Tool_Frame = LabelFrame(root,text="HSV Scale")
    Resize_Tool_Frame = LabelFrame(root,text="Resize")
    Rotate_Tool_Frame = LabelFrame(root,text="Rotate")
    Noise_Tool_Frame = LabelFrame(root,text="Add Noise")
    Filter_Tool_Frame = LabelFrame(root,text="Add Filter")
    Blend_Tool_Frame = LabelFrame(root,text="Blend") 
    
def open_file():
    global panelA, current_img, original_img, buf, dim,masking_img_trigger
    global add_enable,sub_enable,Blend_enable,gray_flag
    global Salt_Pepper_flag,Gaussian_flag,invert_flag 
    global Hist_flag,filter1_state,filter2_state,filter3_state
    global filter4_state,filter5_state,masking_img_trigger,masking_enable,stored_flag,whitebalance_enable
    # file type
    filetypes = (
        ('JPG', '*.jpg'),
        ('All files', '*.*')
    )
    # show the open file dialog
    path = fd.askopenfilename(filetype=filetypes)
    masking_img_trigger = 0
    # show Image
    if len(path) != 0  :
        print(type(path))
        print(path)
        add_enable = 0
        sub_enable = 0
        Blend_enable = 0
        gray_flag = 0
        Salt_Pepper_flag = 0
        Gaussian_flag = 0
        invert_flag = 0 
        Hist_flag = 0
        filter1_state = 0
        filter2_state = 0
        filter3_state = 0
        filter4_state = 0
        filter5_state = 0
        masking_img_trigger = 0
        masking_enable = 0
        stored_flag= 0
        whitebalance_enable = 0
        Hist_garph = 0
        img = cv.imread(path)
        current_img = img.copy()
        original_img = img.copy()
        width_value = original_img.shape[1]
        height_value = original_img.shape[0]
        dim =(width_value,height_value)
        display_img = cv.cvtColor(current_img, cv.COLOR_BGR2RGB)
        img_display = Image.fromarray(display_img)
        img_display = ImageTk.PhotoImage(img_display)
        if panelA is None:
            panelA = Label(Image_Frame,image=img_display)
            panelA.image = img_display
            panelA.grid(row = 0, column= 0)
            Label(Image_Frame,text = "Height:" + str(dim[0]) +"px "+"Width:"+ str(dim[1]) + "px" ).grid(row=1, column=0)
            Generate_Button()
        else:
            panelA.configure(image=img_display)
            panelA.image = img_display
            for item in root.winfo_children():
                if type(item) != type(Menu()) and item is not Tool_Frame and item is not Image_Frame:
                    item.destroy()
            Generate_Button()
        buf = current_img.copy()
        #Update_Image(current_img)

def save_file():
    global buf
    cv.imshow("output",buf)
    output_img=buf
    filetypes = (
        ('JPG', '*.jpg'),
        ('All files', '*.*')
    )
    path =fd.asksaveasfilename(filetypes=filetypes,defaultextension = filetypes)
    if path:
        cv.imwrite(path,output_img)


def funcBrightContrast(x) :
    brightness = Brightness.get()
    contrast = Contrast.get()
    apply_brightness_contrast(brightness,contrast)
    
    
def apply_brightness_contrast(brightness = 255, contrast = 127):#calculate brightness and contrast
    global alpha_b, gamma_b,alpha_c,gamma_c
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
    else:
        alpha_b=1.0
        gamma_b=0.0
    if contrast != 0:
        f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
    else:
        alpha_c=1.0
        gamma_c=0.0
    Update_Image(current_img)

def RGB_Receive(x):
    global R_Value,G_Value,B_Value
    R_Value =R.get()
    G_Value =G.get()
    B_Value =B.get()
    Update_Image(current_img)
    
def HSV_Receive(x):
    global H_Value,S_Value,V_Value
    H_Value =H.get()
    S_Value =S.get()
    V_Value =V.get()
    Update_Image(current_img)
# open file button

Menu_Bar =Menu(root)
root.config(menu=Menu_Bar)

File_Menu = Menu(Menu_Bar)
Menu_Bar.add_cascade(label ="File", menu=File_Menu)
File_Menu.add_command(label="Open...", command= open_file)
File_Menu.add_command(label="Save As...", command= save_file)

Image_Frame = LabelFrame(root,text="Image",bg="white")
Image_Frame.place(relx=0.5,rely=0.5,anchor=CENTER)
Tool_Frame = LabelFrame(root,text="Tool",bg="white")
Tool_Frame.place(relx=0.01,rely=0.5,anchor=W)



root.mainloop()
