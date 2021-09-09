#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vispy: gallery 2, testskip
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# Extened by Todor Arnaudov, 25-11-2018 
# -- Current shadertoy variables do not match, need adjustment
#... added Tiny Clouds webGL 1.0 port
# Select shaders ... 
#  print framerate and viewport
# Frame rate seems limited to 60 fps
# s6.py -- Video, ... images animated, save ... #3-8-2019 
# Но не ще да чете кадри с опенсв в движение от on_timer?
# - затова ги чета в списък и подавам от там

"""
4oy demo. You can copy-paste shader code from an example on
www.shadertoy.com and get the demo.

TODO: support cubes and videos as channel inputs (currently, only images
are supported).

"""

# NOTE: This example throws warnings about variables not being used;
# this is normal because only some shadertoy examples make use of all
# variables, and the GPU may compile some of them away.

import sys
from datetime import datetime, time

import numpy as np
from vispy import gloo
from vispy import app
import vispy.io as io
 
import time as timeOne
import imageio

print(dir(timeOne))
a = timeOne.time()
b = timeOne.time()
print("time()?", b - a)

frame_divider = 10 #+++

vertex = """
#version 120

attribute vec2 position;
void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

fragment = """
#version 120

uniform vec3      iResolution;           // viewport resolution (in pixels)
uniform float     iGlobalTime;           // shader playback time (in seconds)
uniform float     iTime;
uniform vec4      iMouse;                // mouse pixel coords
uniform vec4      iDate;                 // (year, month, day, time in seconds)
uniform float     iSampleRate;           // sound sample rate (i.e., 44100)
uniform sampler2D iChannel0;             // input channel. XX = 2D/Cube
uniform sampler2D iChannel1;             // input channel. XX = 2D/Cube
uniform sampler2D iChannel2;             // input channel. XX = 2D/Cube
uniform sampler2D iChannel3;             // input channel. XX = 2D/Cube
uniform vec3      iChannelResolution[4]; // channel resolution (in pixels)
uniform float     iChannelTime[4];       // channel playback time (in sec)

%s
"""


def get_idate():
    now = datetime.now()
    utcnow = datetime.utcnow()
    midnight_utc = datetime.combine(utcnow.date(), time(0))
    delta = utcnow - midnight_utc
    return (now.year, now.month, now.day, delta.seconds)


def noise(resolution=64, nchannels=1):
    # Random texture.
    return np.random.randint(low=0, high=256,
                             size=(resolution, resolution, nchannels)
                             ).astype(np.uint8)


import cv2

class Canvas(app.Canvas):

    #mSlowDown = 1
    def __init__(self, shadertoy=None, slowDown = 1.0, saveVideo=0, texture=0):
               
        global mReadVideo
        self.cap = None
        global cap
        global video #imageio
        self.cap = 1
                
        if mReadVideo: 
          ret, frame = cap.read(0) #self.frame) #global
          if ret: cv2.imshow("cap", frame) 
          #video = imageio.read('/path/to/movie.mp4')
        # somwhere in a timer callback
          #pass #use global cap
          #self.cap = cv2.VideoCapture(mVideoPath)
          #self.cap.set(cv2.CAP_PROP_POS_FRAMES, 900) #etc.
          #cap = cv2.VideoCapture(mVideoPath)
          #self.cap.set(cv2.CAP_PROP_POS_FRAMES, 900) #etc.
        print(self.cap)
        
        app.Canvas.__init__(self, keys='interactive')        
        self.texture = texture
        if shadertoy is None:
            shadertoy = """
            void main(void)
            {
                vec2 uv = gl_FragCoord.xy / iResolution.xy;
                gl_FragColor = vec4(uv,0.5+0.5*sin(iGlobalTime),1.0);
            }"""
        self.save = saveVideo  
        if texture:
            # Create program
          self.program = gloo.Program(VERT_SHADER, fragment % shadertoy)
                # Set uniforms and samplers
          self.program['a_position'] = gloo.VertexBuffer(positions) #VertexBuffer(positions)
          self.program['a_texcoord'] = gloo.VertexBuffer(texcoords) #VertexBuffer(texcoords)
          #
          self.program['u_texture1'] = gloo.Texture2D(im1)
          self.program['u_texture2'] = gloo.Texture2D(im2)
          self.program['u_texture3'] = gloo.Texture2D(im3)
        else:                    
          self.program = gloo.Program(vertex, fragment % shadertoy)
          
        self.mSlowDown = slowDown; #adjust iTime when rendering is slow 30-7-2019

        self.program["position"] = [(-1, -1), (-1, 1), (1, 1),
                                    (-1, -1), (1, 1), (1, -1)]
                                    
        self.program['iMouse'] = 0, 0, 0, 0

        self.program['iSampleRate'] = 44100.
        for i in range(4):
            self.program['iChannelTime[%d]' % i] = 0.
        self.program['iGlobalTime'] = 0.
        self.activate_zoom()
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)        
        self.frame = 0 #+++     
        self.frameTime = [] #+++ push tuples (frame, time), compute framerate        
        self.show()
        self.startTime = timeOne.time() #timeOne.time() #different than the other datetime! 5-8-2019

    def set_channel_input(self, img, i=0):
        tex = gloo.Texture2D(img)
        tex.interpolation = 'linear'
        tex.wrapping = 'repeat'
        self.program['iChannel%d' % i] = tex
        self.program['iChannelResolution[%d]' % i] = img.shape

    def on_draw(self, event):
        self.program.draw()

    def on_mouse_click(self, event):
        # BUG: DOES NOT WORK YET, NO CLICK EVENT IN VISPY FOR NOW...
        imouse = event.pos + event.pos
        self.program['iMouse'] = imouse

    def on_mouse_move(self, event):
        if event.is_dragging:
            x, y = event.pos
            px, py = event.press_event.pos
            imouse = (x, self.size[1] - y, px, self.size[1] - py)
            self.program['iMouse'] = imouse

    def force_timer(self, elapsed): #DONT CALL
        global mReadImages, mReadVideo, cap
        global frameArr, mFrames
        
        if self.texture:   #Change on each frame    3-8-2019          
          if mReadImages:
            print("self.texture?, frame", self.frame)
            path = mRootPath + str(self.frame%8)+".png"
            print(path)
            #im = io.read_png(path) #mRootPath + str(self.frame%8)+".png")          
            im = io.imread(path) #mRootPath + str(self.frame%8)+".png")          
            #imread requires PIL or imageio
            #directly write video frames from opencv numpy arrays?
           
          if mReadVideo: #read video
            
            try:
              if (self.cap):
                #cv2.waitKey(35)
                print("mReadVideo?")  
                #cap.set(cv2.CAP_PROP_POS_FRAMES, 0)        
                #ret, frame = self.cap.read(0)
                #ret, frame = self.cap.read(self.frame)
                ret, frame = cap.read() #self.frame) #global
                if ret: cv2.imshow("cap", frame)            
                if ret: self.program['u_texture1'] = gloo.Texture2D(frame)
                else: 
                  print("ret=?, frame=?", ret, frame)
                  self.program['u_texture1'] = gloo.Texture2D(frameArr[self.frame%mFrames])
            except(e): print(e)
            
        #print("slef.save=", self.save)
        #sleep(5)
        if self.save: #True: #self.save==1:          
          print("IN slef.save=", self.save)
          i = self.render() #without OK?
          #io.write_png(str(self.frame)+".png", i);
          io.write_jpg(str(self.frame)+".jpg", i);
          #no such: io.write_jpg(str(self.frame)+".jpg", i);
          
        #self.mSlowDown = 1; #if not self.mSlowDown: self.mSlowDown = 1
        self.program['iTime'] = elapsed * self.mSlowDown; # event.elapsed * self.mSlowDown;
        self.program['iGlobalTime'] = elapsed; #event.elapsed
        self.program['iDate'] = get_idate()  # used in some shadertoy exs
        #self.update()
        self.frame+=1  #24-11-2018
        print(self.frame)
        self.frameTime.append((self.frame, elapsed)) #event.elapsed))
        if (self.frame%frame_divider ==0):
          ln = len(self.frameTime);
          fr1,t1 = self.frameTime[ln-1]
          fr2,t2 = self.frameTime[ln-2]
          fps = (fr1-fr2)/(t1-t2)
          #print({:04.2f} fps, end=", ");
          print(" %0.2f"% fps, end=", ");
          sys.stdout.flush()
        
        self.update()
        
    def on_timer(self, event):
        global mReadImages, mReadVideo, cap
        global frameArr, mFrames
        global mOpencvVideo
        
        
        print("(##"+str(event.elapsed)+")");
        if self.texture:   #Change on each frame    3-8-2019          
          if mReadImages:
            print("self.texture?, frame", self.frame)
            path = mRootPath + str(self.frame%8)+".png"
            print(path)
            #im = io.read_png(path) #mRootPath + str(self.frame%8)+".png")          
            im = io.imread(path) #mRootPath + str(self.frame%8)+".png")          
            #imread requires PIL or imageio
            #directly write video frames from opencv numpy arrays?
            self.program['u_texture1'] = gloo.Texture2D(im)
          if mReadVideo: #read video
            if mOpencvVideo:
              try:              
                  if (self.cap):
                    #cv2.waitKey(35)
                    print("mReadVideo?")  
                    #cap.set(cv2.CAP_PROP_POS_FRAMES, 0)        
                    #ret, frame = self.cap.read(0)
                    #ret, frame = self.cap.read(self.frame)
                    ret, frame = cap.read() #self.frame) #global
                    if ret: cv2.imshow("cap", frame)            
                    if ret: self.program['u_texture1'] = gloo.Texture2D(frame)
                    else: 
                      print("ret=?, frame=?", ret, frame)
                      self.program['u_texture1'] = gloo.Texture2D(frameArr[self.frame%mFrames])
                  
              except(e): print(e)
            else:
                 print("IO VIDEO?")
                 im = video.get_next_data()                 
                 im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                 tx = gloo.Texture2D(im)                 
                 cv2.imshow("IO", im)
                 #tx.set_data(video.get_next_data())
                 self.program['u_texture1'] = tx
                 #self.program['u_texture1'] = gloo.Texture2D(video.get_next_data())
            
        #print("slef.save=", self.save)
        #sleep(5)
        if self.save: #True: #self.save==1:          
          print("IN slef.save=", self.save)
          i = self.render() #without OK?
          io.write_png(str(self.frame)+".png", i);
          #no such: io.write_jpg(str(self.frame)+".jpg", i);
          
        print("("+str(event.elapsed)+")");
        #self.mSlowDown = 1; #if not self.mSlowDown: self.mSlowDown = 1
        self.program['iTime'] = event.elapsed * self.mSlowDown;
        self.program['iGlobalTime'] = event.elapsed
        self.program['iDate'] = get_idate()  # used in some shadertoy exs
        #self.update()
        self.frame+=1  #24-11-2018
        print(self.frame)
        self.frameTime.append((self.frame, event.elapsed))
        if (self.frame%frame_divider ==0):
          ln = len(self.frameTime);
          fr1,t1 = self.frameTime[ln-1]
          fr2,t2 = self.frameTime[ln-2]
          fps = (fr1-fr2)/(t1-t2)
          #print({:04.2f} fps, end=", ");
          print(" %0.2f"% fps, end=", ");
          sys.stdout.flush()
        
        self.update()

    def on_resize(self, event):
        self.activate_zoom()

    def activate_zoom(self):
        gloo.set_viewport(0, 0, *self.physical_size)
        self.program['iResolution'] = (self.physical_size[0],
                                       self.physical_size[1], 0.)
        print("WxH: %d %d" % (self.physical_size[0],self.physical_size[1]))
                                      

# -------------------------------------------------------------------------
# COPY-PASTE SHADERTOY CODE BELOW
# -------------------------------------------------------------------------
SHADERTOY = """

/* WebGL 1.0 port of stubbe's: (for-loop transformation)
 https://www.shadertoy.com/view/lsBfDz
"Two tweet (280 chars) reinterpretation of iq's iconic clouds shader 
(https://www.shadertoy.com/view/XslGRr)

Note: 2e2 = 200. BTW, the usage of e-style numbers is more reasonable for 4+ digits:
                 2e3, 2e6 etc. = 2000, 2000000
*/

//#define T texture(iChannel0,(s*p.zw+ceil(s*p.x))/2e2).y/(s+=s)*4.

#define T texture2D(iChannel0,(s*p.zw+ceil(s*p.x))/2e2).y/(s+=s)*4. //YES!!! fixed it!!!


#define MAXSTEPS 202

//void mainImage(out vec4 O,vec2 x){
void main(){ //out vec4 O,vec2 gl_FragCoord){
    vec4 p=vec4(.8,0,gl_FragCoord/iResolution.y-.8);
    vec4 d = p;
    vec4 c=vec4(.6,.7,d);
    gl_FragColor=c-d.w;
    float f,s,t=2e2+sin(dot(gl_FragCoord,gl_FragCoord));
    t--;           
    for(int i=0; i<MAXSTEPS; i++) {
        p=.05*t*d,
        p.xz+=iGlobalTime,       
        s=2.,
        f=p.w+1.-T-T-T-T,
    	f<0.?gl_FragColor+=(gl_FragColor-1.-f*c.zyxw)*f*.4:gl_FragColor;
        t--;       
    }
    
}

"""


shaders = []

shaders.append("""
/*
  ["Hackafe Logo"] Exercise #1. Version: 27.12.2017
  https://www.shadertoy.com/view/4lffzf
  [Author]: Todor "Tosh" Arnaudov (Twenkid) http://artificial-mind.blogspot.bg | http://research.twenkid.com
  [Credits and Thanks]: Shadertoy community, iq, Dave Hoskins, rear, LeGuignon; BigWings; Phong; CG pioneers and mathematicians from the past 
                        "Hackafe" existed in Plovdiv, Bulgaria. Currently it's hibernated: http://www.hackafe.org/                 
  [Story]
  Hackafe - "The Plovdiv's Hackerspace" - was established in 2013 A.D. with love and enthusiasm.
  It had a youth of passion and sailed in an ocean of expectations, but the Water happened to be
  too deep and stormy, thus the cruise was quite short.
  The sailors and captains weren't strong enough to sustain...
  Their relationships and activities became increasingly dysfunctional.
  The atmosphere went worse and worse, and the hackerspace irreversibly declined
  to its shameful oblivion and death.
  After a long painful agony the last survivors put Hackafe to sleep in October 2017.
  
  A few genes lasted, though. They were launched deep into the Cyberspace 
  to find a better planet where to bloom and live again...   
  [Future work:]
      1. Antialiasing and motion blur
      2. More spatial artifacts in the background (specfic stars/larger variety of brightness/color, comets, planets, black hole, asteroids flying)
      3. Cloud/fog
      4. Icy/semi-transparent "bumpy" blocks (like for example the IcePrimitives shader) to exercise refraction, subsurface scattering
      5. More freedom for the blocks - more rotations, phsysics, bouncing; interactivity through the mouse, hitting by asteroids, laser beams
      6. Electricity arcs around the blocks, lightning strikes
      7. Raining lava?, blocks reacting to the hits - heating/cooling ...
      8. More complex and varying sound
      9. More complex lighting, Fresnel equations, shadows, ...
      10. Story, travelling, scene changes, action
                        
*/

float z = 0.05;
float StepDiv = 35.;
float StepBase = 0.30; //Blocks
const float cube = 3.; //(~) distance to distinguish the cube from space
float step = 0.8; //Blocks
const vec2 cameraYz = vec2(2.65,-5.5); //the X is animated
float cameraSpeed = 2.4; //2.7;

const int STEPS = 50; //ray marching steps
const float EPS = 0.001; //precision (epsilon)

// from iq's "Anels", from Kali's Lonely Tree shader, from Analytical geometry textbooks - rotation around axis
mat3 rotationMat(in vec3 v, in float angle) //, in out vec3 vOut)
{
    float c = cos(angle), s = sin(angle);
    return mat3(c + (1.0 - c) * v.x * v.x, 
                (1.0 - c) * v.x * v.y - s * v.z,
                (1.0 - c) * v.x * v.z + s * v.y,
                
                (1.0 - c) * v.x * v.y + s * v.z,
                c + (1.0 - c) * v.y * v.y,
                (1.0 - c) * v.y * v.z - s * v.x,
                
                (1.0 - c) * v.x * v.z - s * v.y,
                (1.0 - c) * v.y * v.z + s * v.x,
                c + (1.0 - c) * v.z * v.z);
}

// jerome, Electricity
float rand(vec2 n) {
    return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}

float noise(vec2 n) {
    const vec2 d = vec2(0.0, 1.0);
    vec2 b = floor(n), f = smoothstep(vec2(0.0), vec2(1.0), fract(n));
    return mix(mix(rand(b), rand(b + d.yx), f.x), mix(rand(b + d.xy), rand(b + d.yy), f.x), f.y);
}

float fbm8(vec2 n) {
    float total = 0.0, amplitude = 1.0;
    for (int i = 0; i < 8; i++) {
        total += noise(n)/2. * amplitude;
        n += n;
        amplitude *= 0.5;
    }
    return total;
}

//Based on Electricity, but modified by Twenkid to look somewhat like a burst of hot gas.
//Should be optimized and varied - e.g. arrays and cycles/more streams/more adjustible
vec3 electricity(vec2 uv ){    
   vec2 t = uv * vec2(2.0,1.0) - iGlobalTime*4.0;  //iTime = iGlobalTime    
   float ybase = 0.30, ystep = 0.03;
   float ycenter = ybase+fbm8(t)*0.35;
    
   float ycenter2 = ybase+ystep+fbm8(t)*0.5;
   float ycenter3 = ybase-ystep+fbm8(t)*0.5;    
   float diff = abs(uv.y - ycenter);
   float c1 = 1.0 - mix(0.0,0.3,diff*21.0);
     
    c1 = clamp(c1, 0., 1.);
    vec3 col = vec3(c1*0.9, 0.9*c1,c1*0.2);
        
    float diff2 = abs(uv.y - ycenter2);
    float c2 = 1.0 - mix(0.0,0.2,diff2*21.0);    
    col = mix(col, vec3(c2*0.7, 0.4*c2, c2*0.1), 0.7);
    
    float d3 = abs(uv.y - ycenter3);
    float c3 = 1.0 - mix(0.0,0.3,diff2*21.0);
    col = mix(col, vec3(c3*0.5, 0.3*c3, c3*0.1), 0.5);
    //col = mix(col, vec3(c3*0.7+abs(noise(uv)/5.), 0.3*c3, c3*0.1), 0.5); //noise - no, too jaggy
   // col = min(col, vec3(c3*0.7+abs(fbm8(uv)/5.), 0.3*c3, c3*0.1));
    col = max(vec3(0.), col); //avoid negative color - electricity is multiplied in the render
    return col;
}
// jerome's end

//// Dave Hoskins's? noise
float N1(float t) { return fract(sin(t*10234.324)*123423.23512);  }

vec2 N22(vec2 p)
{	// Dave Hoskins - https://www.shadertoy.com/view/4djSRW   //modified to vec2, vec2
	vec2 p2  = fract(vec2(p.xyx) * vec2(443.897, 441.423)); // 437.195));
    p2 += dot(p2, p2.yx + 19.19);
    return fract(vec2((p2.x + p2.y)*p2.x, (p2.y+p2.y)*p2.x));
}

vec2 fbm(vec2 v){ return N22(v)*0.5 + vec2(0.25*N1(v.x)) + vec2(0.25*N1(v.y)); }

//iq's box
float sdBox( vec3 p, vec3 b ) { // float zoom = 3.;   
    vec3 d = (abs(p) - b);
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}
//float sdPlane(vec3 p){return p.y;} //future use - grid plane?

float distLimitBorder(vec3 r){ //26-12-2017
   //float step = StepBase + sin(iGlobalTime)/STEPDIV;
   //float d3 = sdBox(r-vec3(1.0,0.25, 0.0), vec3(0.1, 0.1, z));
   //return d3;
    
   vec3 axis = normalize(vec3(1.0, 0.25, 0.0));  
   r*= rotationMat(axis, mod(fract(iGlobalTime)*6.28, 6.28));    
   float d1 = sdBox(r-vec3( 1.0,0.25, 0.0), vec3(0.1, 0.1, z));  
   return d1;    
}  
/*
float distLimitBig(vec3 r){ //26-12-2017 - right border/curtain
   float step = StepBase+ sin(iGlobalTime)/StepDiv;
   float d3 = sdBox(r-vec3(10.0,StepBase, 0.0), vec3(9., 10., z));
   return d3;
}
*/
  
//Distance from the objects. Should be optimized, could use one or a few common formulas,
//except for the rotating block.
float dist(vec3 r)
{      
   float step = StepBase + sin(iGlobalTime)/StepDiv;           
    vec3 axis = normalize(vec3(1.0, 0.25, 0.0));     
    vec3 r1 = r * rotationMat(axis, mod(fract(iGlobalTime)*6.28, 6.28));    
    float d1 = sdBox(r1-vec3( 1.0,0.25, 0.0), vec3(0.1, 0.1, z));  
    
    float d2 = sdBox(r-vec3(1.0-step,0.25, 0.0), vec3(0.1, 0.1, z));
    float d3 = sdBox(r-vec3(1.0-step-step,0.25, 0.0), vec3(0.1, 0.1, z));
        
    float d4 = sdBox(r-vec3(1.0,0.25+step, 0.0), vec3(0.1, 0.1, z));
    float d5 = sdBox(r-vec3(1.0-step,0.25+step, 0.0), vec3(0.1, 0.1, z));
    float d6 = sdBox(r-vec3(1.0-step-step,0.25+step, 0.0), vec3(0.1, 0.1, z));
    
    float d7 = sdBox(r-vec3(1.0,0.25-step, 0.0), vec3(0.1, 0.1, z));   
    float d8 = sdBox(r-vec3(1.0-step-step,0.25-step, 0.0), vec3(0.1, 0.1, z));
    
    float d = min(d1,d2);
    d = min(d, min(d3,d4));
    d = min(d, min(d5,d6));
    d = min(d, min(d7,d8));       
    return d;
}

// Normal vector - http://www.pouet.net/topic.php?which=7920&page=10 by rear
vec3 normal(vec3 p)
{
	#define dr 1e-5
	vec3 drx = vec3(dr,0,0);
	vec3 dry = vec3(0,dr,0);
	vec3 drz = vec3(0,0,dr);
	return ( vec3( dist(p+drx), dist(p+dry), dist(p+drz) ) - dist(p)) / dr;
}

void main(void) //( out vec4 fragColor, in vec2 fragCoord )
//void main(out vec4 gl_FragColor, in vec2 gl_FragCoord )
{        
	vec2 uv = gl_FragCoord.xy / iResolution.xy;
    gl_FragColor = vec4(0.);    
    //gl_FragColor = vec4(uv, uv);     return; test
    vec2 r = (gl_FragCoord.xy / iResolution.xy);
	r.x*=(iResolution.x/iResolution.y);	           
    r -=vec2(0.1, 0.8);    		    
    vec3 camera = vec3(1.05+(sin(iGlobalTime))*cameraSpeed,cameraYz); //more to the center      	
    vec3 ro =  vec3(r.x, r.y+1.0, -1.0);       
    vec3 p = ro;  //ray origin          
	vec3 dir = normalize(p-camera); //ray direction
    float d; //distance
    
	for(int i=0; i<STEPS; i++) //Ray marching
	{
		d = dist(p);
		if(d < EPS) break;
		p = p+dir*d;
	}

    vec3 materialcolor=vec3(0.);        
    int m;  
    if (d<=cube) { m = 0; materialcolor = vec3(0.9,.9,.2);} //cube;
    else { m = 1; } //materialcolor = vec3(0.);}
            
	vec3 nor = normal(p);  // normal vector
    vec3 lightpos = vec3(1.5-sin(iGlobalTime)*5., 0.1+sin(iGlobalTime), 3.5+sin(iGlobalTime)*5.);           
    lightpos.y +=sin(iGlobalTime); // [-1., +1]
         
    vec3 lightdir = normalize(vec3(0.3,0.3,0.3)-lightpos);
   	
    float light = 1.0 + 0.01*(dot(nor,lightpos)); //intensity
    
    light *=  pow(dist(lightdir-p), 2.);
    
	//vec3 color = vec3(light);
    vec3 color = vec3(1.0-light/5.); //vec3(1.0, 1.0, 1.0);
    color = clamp( materialcolor*color, 0., 1.0);
    
    //Phong
    float dif = clamp( dot( nor, lightdir ), 0.0, 1.0 ); //iq diffuse
    vec3  ref = reflect( dir, nor );  //reflection
    float spe = pow(clamp( dot( ref, lightdir ), 0.0, 1.0 ),16.0); //specular component
        
    color+=dif/3. + spe/2.;
        
	gl_FragColor = vec4(color, 1.0);
    gl_FragColor.xyz = vec3(color);
    gl_FragColor.w = m ==0 ? 1. : 0.; //
             
    vec2 pos = 2.0 * vec2(gl_FragCoord.xy - 0.5 * iResolution.xy) / iResolution.y;
    
   //The space, stars... #28-11-2017 & electricity
   if ( gl_FragColor.r < 0.001 && m==1)
    {     
        //To do: add more effects: specific stars, nebullas, planets, comets, black hole etc.... had meteors, but were removed;    
        vec2 n1 = N22(pos);
        float star = n1.x < 0.07 ? 0.1 : 0.;        
        star+= n1.y > 0.97 ? N1(n1.x)/1.0*(max(star, sin(iGlobalTime))) : 0.0;
        vec2 fb = fbm(pos);
        star*=max(fb.x, fb.y);       
        gl_FragColor += star*1.4; //brighter stars
        gl_FragColor.a = 1.0;                                 
        gl_FragColor.xyzw+=vec4(clamp(abs(cos(iGlobalTime/3.5))*4.28, 1., 3.)*electricity(uv), 1.0); //thick line        
    }
    else //The blocks
    {     
     const float EPSLIMIT = EPS*5.; //0.06;
     float limit = 0.0;     
     ro =  vec3(r.x, r.y+1.0, -1.0);    
     p = ro;
	 dir = normalize(p-camera);                     
	 for(int i=0; i<STEPS/3; i++)  //Second marching for the rotating block and the burst. Fewer steps and lower precision are enough.
	 {
        d = distLimitBorder(p);
		if(d < EPSLIMIT) break;
		p = p + dir * d;
	  }
        
      if (d<EPSLIMIT) gl_FragColor.xyz += electricity(uv);
            
      gl_FragColor.w = 1.0; //Alpha
      
    }
   
    //Gamma correction
     gl_FragColor.xyz=pow(gl_FragColor.xyz, vec3(1.4));
    
}
""");


shaders.append("""

/* WebGL 1.0 port of stubbe's: (for-loop transformation)
 https://www.shadertoy.com/view/lsBfDz
"Two tweet (280 chars) reinterpretation of iq's iconic clouds shader 
(https://www.shadertoy.com/view/XslGRr)

Note: 2e2 = 200. BTW, the usage of e-style numbers is more reasonable for 4+ digits:
                 2e3, 2e6 etc. = 2000, 2000000
*/

//#define T texture(iChannel0,(s*p.zw+ceil(s*p.x))/2e2).y/(s+=s)*4.

#define T texture2D(iChannel0,(s*p.zw+ceil(s*p.x))/2e2).y/(s+=s)*4. //YES!!! fixed it!!!


#define MAXSTEPS 202

//void mainImage(out vec4 O,vec2 x){
void main(){ //out vec4 O,vec2 gl_FragCoord){
    vec4 p=vec4(.8,0,gl_FragCoord/iResolution.y-.8);
    vec4 d = p;
    vec4 c=vec4(.6,.7,d);
    gl_FragColor=c-d.w;
    float f,s,t=2e2+sin(dot(gl_FragCoord,gl_FragCoord));
    t--;           
    for(int i=0; i<MAXSTEPS; i++) {
        p=.05*t*d,
        p.xz+=iGlobalTime,       
        s=2.,
        f=p.w+1.-T-T-T-T,
    	f<0.?gl_FragColor+=(gl_FragColor-1.-f*c.zyxw)*f*.4:gl_FragColor;
        t--;       
    }
    
}

""");

#NOISE
shaders.append("""
void main(){} 
""");



#Sacred computer, added 30.7.2019
shaders.append("""

float z = 0.015;//5;
float StepDiv = 35.;
float StepBase = 0.30; //Blocks
const float cube = 3.; //(~) distance to distinguish the cube from space
float step = 0.8; //Blocks
const vec2 cameraYz = vec2(2.5,-6.5); //vec2(2.65,-5.5); //the X is animated
float cameraSpeed = 1.5; //2.4; //2.7;

const int STEPS = 50; //ray marching steps
const float EPS = 0.001; //precision (epsilon)
const float PI = 3.1415926;

// from iq's "Anels", from Kali's Lonely Tree shader, from Analytical geometry textbooks - rotation around axis
mat3 rotationMat(in vec3 v, in float angle) //, in out vec3 vOut)
{
    float c = cos(angle), s = sin(angle);
    return mat3(c + (1.0 - c) * v.x * v.x, 
                (1.0 - c) * v.x * v.y - s * v.z,
                (1.0 - c) * v.x * v.z + s * v.y,
                
                (1.0 - c) * v.x * v.y + s * v.z,
                c + (1.0 - c) * v.y * v.y,
                (1.0 - c) * v.y * v.z - s * v.x,
                
                (1.0 - c) * v.x * v.z - s * v.y,
                (1.0 - c) * v.y * v.z + s * v.x,
                c + (1.0 - c) * v.z * v.z);
}

// jerome, Electricity
float rand(vec2 n) {
    return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}

float noise(vec2 n) {
    const vec2 d = vec2(0.0, 1.0);
    vec2 b = floor(n), f = smoothstep(vec2(0.0), vec2(1.0), fract(n));
    return mix(mix(rand(b), rand(b + d.yx), f.x), mix(rand(b + d.xy), rand(b + d.yy), f.x), f.y);
}

float fbm8(vec2 n) {
    float total = 0.0, amplitude = 1.0;
    for (int i = 0; i < 8; i++) {
        total += noise(n)/2. * amplitude;
        n += n;
        amplitude *= 0.5;
    }
    return total;
}


//Based on Electricity, but modified by Twenkid to look somewhat like a burst of hot gas.
//Should be optimized and varied - e.g. arrays and cycles/more streams/more adjustible
vec3 electricity(vec2 uv ){    
   vec2 t = uv * vec2(2.0,1.0) - iTime*4.0;      
   float ybase = 0.30, ystep = 0.03;
   float ycenter = ybase+fbm8(t)*0.35;
    
   float ycenter2 = ybase+ystep+fbm8(t)*0.5;
   float ycenter3 = ybase-ystep+fbm8(t)*0.5;    
   float diff = abs(uv.y - ycenter);
   float c1 = 1.0 - mix(0.0,0.3,diff*21.0);
     
    c1 = clamp(c1, 0., 1.);
    vec3 col = vec3(c1*0.9, 0.9*c1,c1*0.2);
        
    float diff2 = abs(uv.y - ycenter2);
    float c2 = 1.0 - mix(0.0,0.2,diff2*21.0);    
    col = mix(col, vec3(c2*0.7, 0.4*c2, c2*0.1), 0.7);
    
    float d3 = abs(uv.y - ycenter3);
    float c3 = 1.0 - mix(0.0,0.3,diff2*21.0);
    col = mix(col, vec3(c3*0.5, 0.3*c3, c3*0.1), 0.5);
    //col = mix(col, vec3(c3*0.7+abs(noise(uv)/5.), 0.3*c3, c3*0.1), 0.5); //noise - no, too jaggy
   // col = min(col, vec3(c3*0.7+abs(fbm8(uv)/5.), 0.3*c3, c3*0.1));
    col = max(vec3(0.), col); //avoid negative color - electricity is multiplied in the render
    return col;
}
// jerome's end

//// Dave Hoskins's? noise
float N1(float t) { return fract(sin(t*10234.324)*123423.23512);  }

vec2 N22(vec2 p)
{	// Dave Hoskins - https://www.shadertoy.com/view/4djSRW   //modified to vec2, vec2
	vec2 p2  = fract(vec2(p.xyx) * vec2(443.897, 441.423)); // 437.195));
    p2 += dot(p2, p2.yx + 19.19);
    return fract(vec2((p2.x + p2.y)*p2.x, (p2.y+p2.y)*p2.x));
}

vec2 fbm(vec2 v){ return N22(v)*0.5 + vec2(0.25*N1(v.x)) + vec2(0.25*N1(v.y)); }

//iq's box
float sdBox( vec3 p, vec3 b ) { // float zoom = 3.;   
    vec3 d = (abs(p) - b);
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}


float sdBoxAngle( vec3 p, vec3 b, vec3 axis, float angle ) { // float zoom = 3.;      
    axis = normalize(axis);
    p = p*rotationMat(axis, angle);
    float d = sdBox((p-b), b);//(abs(p) - b);
    return d;
    //d = d*rotationMat(d,a);
    //return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

//float sdPlane(vec3 p){return p.y;} //future use - grid plane?

float distLimitBorder(vec3 r){ //26-12-2017
   //float step = StepBase + sin(iTime)/STEPDIV;
   //float d3 = sdBox(r-vec3(1.0,0.25, 0.0), vec3(0.1, 0.1, z));
   //return d3;
    
   vec3 axis = normalize(vec3(1.0, 0.25, 0.0));  
   r*= rotationMat(axis, mod(fract(iTime)*6.28, 6.28));    
   float d1 = sdBox(r-vec3( 1.0,0.25, 0.0), vec3(0.1, 0.1, z));  
   return d1;    
}  
/*
float distLimitBig(vec3 r){ //26-12-2017 - right border/curtain
   float step = StepBase+ sin(iTime)/StepDiv;
   float d3 = sdBox(r-vec3(10.0,StepBase, 0.0), vec3(9., 10., z));
   return d3;
}
*/
  
//Distance from the objects. Should be optimized, could use one or a few common formulas,
//except for the rotating block.
float dist(vec3 r)
{      
    float d9, d10, d11, d12,d13,d14,d15,d16, d17, d18,d19;
   float step = StepBase + sin(iTime)/StepDiv;           
    vec3 axis = normalize(vec3(1.0, 0.25, 0.0));     
    vec3 r1 = r * rotationMat(axis, mod(fract(iTime)*6.28, 6.28)); 
    float d;
    float d1,d2,d3,d4,d5,d6,d7,d8;
    float d20,d21,d22,d23,d24,d25,d26,d27,d28,d29,d30,d31,d32,d33,d34,d35,d36,d37,d38,d39,d40;
    float d41,d42,d43,d44,d45,d46,d47,d48,d49,d50,d51,d52,d53,d54,d55;
    
    /*
    float d1 = sdBox(r1-vec3( 1.0,0.25, 0.0), vec3(0.1, 0.1, z));      
    float d2 = sdBox(r-vec3(1.0-step,0.25, 0.0), vec3(0.1, 0.1, z));
    float d3 = sdBox(r-vec3(1.0-step-step,0.25, 0.0), vec3(0.1, 0.1, z));
        
    float d4 = sdBox(r-vec3(1.0,0.25+step, 0.0), vec3(0.1, 0.1, z));
    float d5 = sdBox(r-vec3(1.0-step,0.25+step, 0.0), vec3(0.1, 0.1, z));
    float d6 = sdBox(r-vec3(1.0-step-step,0.25+step, 0.0), vec3(0.1, 0.1, z));
    
    float d7 = sdBox(r-vec3(1.0,0.25-step, 0.0), vec3(0.1, 0.1, z));   
    float d8 = sdBox(r-vec3(1.0-step-step,0.25-step, 0.0), vec3(0.1, 0.1, z));
    */
    
    //S    
    d1 = sdBox(r-vec3( 0.2,0.6, 0.0), vec3(0.12, 0.03, z));  //HORIZ    
    d2 = sdBox(r-vec3( 0.09,0.465, 0.0), vec3(0.03, 0.13, z));  //VERT
    d3 = sdBox(r-vec3( 0.2,0.33, 0.0), vec3(0.12, 0.03, z));  //HORIZ
    
    
    //V
    d4 = sdBox(r-vec3( 0.44, 0.5, 0.0), vec3(0.05, 0.01, z));  //HORIZ    
    
    d5 = sdBox(r-vec3( 0.375,0.42, 0.0), vec3(0.02, 0.09, z));  //VERTICAL LEFT
    
    d6 = sdBox(r-vec3( 0.43,0.32, 0.0), vec3(0.075, 0.01, z)); //HORIZ DOWN
    
    d7 = sdBox(r-vec3( 0.43,0.42, 0.0), vec3(0.07, 0.01, z)); //HORIZ MIDDLE
    
    d8 = sdBox(r-vec3( 0.47,0.46, 0.0), vec3(0.02, 0.05, z)); //VERTICAL RIGHT UP
    d9 = sdBox(r-vec3( 0.49,0.37, 0.0), vec3(0.02, 0.04, z)); //VERTICAL RIGHT DOWN    
    
    
    float s = 0.18;
    //E
    d10 = sdBox(r-vec3( 0.44+s, 0.5, 0.0), vec3(0.045, 0.01, z));  //z+ is UP!
    //d2 = sdBox(r-vec3( 0.2,0.6, 0.0), vec3(0.03, 0.12, z));  //z+ is UP! +
    //+
    d11 = sdBox(r-vec3( 0.375+s,0.42, 0.0), vec3(0.02, 0.09, z));  //z+ is UP!    
    d13 = sdBox(r-vec3( 0.435+s,0.41, 0.0), vec3(0.045, 0.01, z));
    d12 = sdBox(r-vec3( 0.415+s,0.32, 0.0), vec3(0.06, 0.01, z));
    
    //Sht
    float st = 0.44+s/2.;
    float s1 = 0.07;
    st+=s;
    
    d14 = sdBox(r-vec3(st ,0.42, 0.0), vec3(0.02, 0.09, z)); //VERT
    d15 = sdBox(r-vec3(st + s1 ,0.42, 0.0), vec3(0.02, 0.09, z)); //VERT
    d16 = sdBox(r-vec3(st + s1*2. ,0.42, 0.0), vec3(0.02, 0.09, z)); //VERT
    d17 = sdBox(r-vec3(st + s1, 0.32, 0.0), vec3(0.075, 0.01, z)); //HORIZ DOWN
    
    d18 = sdBox(r-vec3(st + s1*2. + 0.03, 0.34, 0.0), vec3(0.02, 0.01, z)); //HORIZ DOWN RIGHT
    //d19 = sdBox(r-vec3(st + s1*2. + 0.03 + 0.03, 0.34, 0.0), vec3(0.01, 0.03, z)); //HORIZ DOWN RIGHT - MIDDLE
    d19 = sdBox(r-vec3(st + s1*2. + 0.03 + 0.012, 0.33, 0.0), vec3(0.01, 0.03, z)); //HORIZ DOWN RIGHT - MIDDLE
    
    
    float sm = s + s/2.0 + 0.09 + s + 0.03;
    //E
    d20 = sdBox(r-vec3( 0.44+sm, 0.5, 0.0), vec3(0.045, 0.01, z));  
   
    d21 = sdBox(r-vec3( 0.375+sm,0.42, 0.0), vec3(0.02, 0.09, z));   
    d22 = sdBox(r-vec3( 0.435+sm,0.41, 0.0), vec3(0.045, 0.01, z));
    d23 = sdBox(r-vec3( 0.415+sm,0.32, 0.0), vec3(0.06, 0.01, z));
    
    sm += 0.09;
    //N
    //d24 = sdBox(r-vec3( 0.44+sm, 0.5, 0.0), vec3(0.045, 0.01, z));
    d24 = sdBox(r-vec3( 0.44+sm,0.41, 0.0), vec3(0.02, 0.10, z));
    d25 = sdBox(r-vec3( 0.44+sm+0.09,0.41, 0.0), vec3(0.02, 0.10, z));
    d26 = sdBox(r-vec3( 0.44+sm+0.04,0.41, 0.0), vec3(0.06, 0.01, z));
    
    sm+=0.09+0.075;
    //I
    d27 = sdBox(r-vec3( 0.44+sm,0.41, 0.0), vec3(0.02, 0.10, z));
    d28 = sdBox(r-vec3( 0.44+sm+0.09,0.41, 0.0), vec3(0.02, 0.10, z));
    d29 = sdBox(r-vec3( 0.44+sm+0.04,0.32, 0.0), vec3(0.06, 0.01, z)); //HOR DOWN
    
    sm +=0.09+0.15;
    float cor = 0.05;
    //Ya
    d30 = sdBox(r-vec3( 0.44+sm, 0.5, 0.0), vec3(0.036, 0.01, z));  //HORIZ TOP    
    
    d31 = sdBox(r-vec3( 0.375+sm+0.09,0.41, 0.0), vec3(0.02, 0.10, z));  //VERTICAL RIGHT
    
    //d32 = sdBox(r-vec3( 0.43+sm,0.32, 0.0), vec3(0.075, 0.01, z)); //HORIZ DOWN
    d32 = d31;
    
    d33 = sdBox(r-vec3( 0.41+sm,0.41, 0.0), vec3(0.06, 0.01, z)); //HORIZ MIDDLE
    
    d34 = sdBox(r-vec3( 0.39+sm,0.45, 0.0), vec3(0.013, 0.05, z)); //VERTICAL LEFT UP
    d35 = sdBox(r-vec3( 0.36+sm,0.36, 0.0), vec3(0.02, 0.050, z)); //VERTICAL LEFT DOWN
    float dq1 = min(d30,d31), dq2 = min(d32,d33), dq3 = min(d34,d35);
    dq2 = min(dq2, dq3);
    dq1 = min(dq1, dq2);
    
    //T
    sm+=0.09+0.03;    
    d36 = sdBox(r-vec3( 0.44+sm+0.04,0.41, 0.0), vec3(0.02, 0.10, z)); //VER
    //d37 = sdBox(r-vec3( 0.44+sm+0.09,0.41, 0.0), vec3(0.02, 0.10, z));
    d37 = sdBox(r-vec3( 0.44+sm+0.04,0.49, 0.0), vec3(0.09, 0.02, z)); //HOR MID   
    d = min(d36,d37);
    dq1 = min(dq1,d);
    //dq1 = min(dq1, d38);
    
    float sy = 0.3;
    sm = 0.2;
    //SECOND LINE
    //S   
    /*
    d38 = sdBox(r-vec3( 0.2 + sm,0.5-sy, 0.0), vec3(0.09, 0.015, z));    
    d39 = sdBox(r-vec3( 0.11 + sm,0.41-sy, 0.0), vec3(0.02, 0.09, z));  //z+ is UP!
    d40 = sdBox(r-vec3( 0.2 + sm,0.32-sy, 0.0), vec3(0.09, 0.015, z));  //VERT
    d = min(d38, d39);
    d = min(d, d40);
    dq1 = min(dq1, d);
    */
    d38 = sdBox(r-vec3( 0.2 + sm + 0.02,0.5-sy, 0.0), vec3(0.06, 0.015, z));    
    d39 = sdBox(r-vec3( 0.11 + sm + 0.04,0.41-sy, 0.0), vec3(0.02, 0.09, z)); //VERT
    d40 = sdBox(r-vec3( 0.2 + sm +0.02,0.32-sy, 0.0), vec3(0.06, 0.015, z));  
    d = min(d38, d39);
    d = min(d, d40);
    dq1 = min(dq1, d);
    
    /*
    //M - ne, 16:50
    //Nakloneni linii - ostri. NE, po-dobre pravoagalni kato drugite!
    //vec3(0.0,1.0,0.0)  ---- 
    sm = 0.15;
    float angle = -PI/2.5; // fract(iTime)*6.28;
    d41 = sdBoxAngle(r-vec3( 0.2+sm,-0.07, 0.0), vec3(0.12, 0.015, z), vec3(0.0, 0.0,1.0), angle);  
    d42 = sdBoxAngle(r-vec3( 0.2+sm+0.05,0.15, 0.0), vec3(0.09, 0.015, z), vec3(0.0, 0.0,1.0), -angle);  
    d43 = sdBoxAngle(r-vec3( 0.2+sm+0.14,-0.01, 0.0), vec3(0.09, 0.015, z), vec3(0.0, 0.0,1.0), angle);  
    d44 = sdBoxAngle(r-vec3( 0.2+sm+0.17,0.15, 0.0), vec3(0.11, 0.015, z), vec3(0.0, 0.0,1.0), -angle); 
    //d45 = sdBox(r-vec3( 0.2+sm+0.12,0.0, 0.01), vec3(0.02, 0.015, z));
    d = min(d41,d42);   
    d = min(d, d43);
    d = min(d, d44);
    //d = min(d,d45);
    dq1 = min(d,dq1);
     */
    
    sm += -0.11;
    //M
    d41 = sdBox(r-vec3( 0.44+sm,0.41-sy, 0.0), vec3(0.02, 0.10, z)); //VERT
    d42 = sdBox(r-vec3( 0.44+sm+0.18,0.41-sy, 0.0), vec3(0.02, 0.10, z)); //VERT
    d43 = sdBox(r-vec3( 0.44+sm+0.09,0.41-sy-0.03, 0.0), vec3(0.025, 0.01, z)); //HORIZ down   
    d = min(d41,d42);   
    dq1 = min(dq1, d);
    dq1 = min(dq1,d43);
    d43 = sdBox(r-vec3( 0.44+sm+0.04,0.41-sy+0.07, 0.0), vec3(0.03, 0.01, z)); //HORIZ
    d44 = sdBox(r-vec3( 0.44+sm+0.18-0.04,0.41-sy+0.07, 0.0), vec3(0.03, 0.01, z)); //HORIZ
    dq1 = min(dq1,d43);
    dq1 = min(dq1,d44);
    
    d41 = sdBox(r-vec3( 0.44+sm+0.065,0.41-sy+0.02, 0.0), vec3(0.015, 0.04, z)); //VERT
    d42 = sdBox(r-vec3( 0.44+sm+0.18-0.065,0.41-sy+0.02, 0.0), vec3(0.015, 0.04, z)); //VERT
    dq1 = min(dq1,d41);
    dq1 = min(dq1,d42);

    
    //E
    sm+=0.32;
    d40 = sdBox(r-vec3( 0.44+sm, 0.5-sy, 0.0), vec3(0.05, 0.01, z));  
   
    d41 = sdBox(r-vec3( 0.375+sm,0.42-sy, 0.0), vec3(0.02, 0.09, z));   
    d42 = sdBox(r-vec3( 0.435+sm,0.41-sy, 0.0), vec3(0.045, 0.01, z));
    d43 = sdBox(r-vec3( 0.415+sm,0.32-sy, 0.0), vec3(0.06, 0.01, z));
    d = min(dq1, d40);
    d = min(d, d41);
    d = min(d,d42);
    d = min(d,d43);
    dq1 = min(d, dq1);
    
    //T
    sm+=0.13;    
    d36 = sdBox(r-vec3( 0.44+sm+0.04,0.41-sy, 0.0), vec3(0.02, 0.10, z)); //VER
    //d37 = sdBox(r-vec3( 0.44+sm+0.09,0.41, 0.0), vec3(0.02, 0.10, z));
    //d37 = sdBox(r-vec3( 0.44+sm+0.04,0.49-sy, 0.0), vec3(0.09, 0.02, z)); //HOR MID THICK 0.02
    d37 = sdBox(r-vec3( 0.44+sm+0.04,0.49-sy+0.01, 0.0), vec3(0.09, 0.015, z)); //HOR MID THINNER 0.015
    d = min(d36,d37);
    dq1 = min(dq1,d);
    
    //A    
    sm += 0.24;        
    d30 = sdBox(r-vec3( 0.44+sm-0.03, 0.5-sy, 0.0), vec3(0.035, 0.01, z));  //HORIZ TOP    
    d34 = sdBox(r-vec3( 0.39+sm-0.02,0.45-sy, 0.0), vec3(0.013, 0.05, z)); //VERTICAL LEFT UP
    
    //d31 = sdBox(r-vec3( 0.375+sm+0.09,0.41-sy, 0.0), vec3(0.02, 0.10, z));  //VERTICAL RIGHT //LONG for Ya
    d31 = sdBox(r-vec3( 0.375+sm+0.09, 0.41-sy-0.05, 0.0), vec3(0.02, 0.05, z));  //VERTICAL RIGHT DOWN //LONG for Ya        
    d36 = sdBox(r-vec3( 0.375+sm+0.09-0.01, 0.41-sy+0.05, 0.0), vec3(0.013, 0.04, z)); //VERTICAL RIGHT UP
    
    //d32 = sdBox(r-vec3( 0.43+sm,0.32, 0.0), vec3(0.075, 0.01, z)); //HORIZ DOWN
    d32 = d31;
    
    d33 = sdBox(r-vec3( 0.41+sm+0.005,0.41-sy, 0.0), vec3(0.065, 0.01, z)); //HORIZ MIDDLE    
    d35 = sdBox(r-vec3( 0.36+sm,0.36-sy, 0.0), vec3(0.02, 0.050, z)); //VERTICAL LEFT UP
    
    dq1 = min(dq1, d30);
    dq2 = min(d31,d32); dq3 = min(d33,d34);
    dq1 = min(dq1,dq2);
    dq1 = min(dq1,dq3);
    dq1 = min(dq1,d35);
    dq1 = min(dq1,d36);
           
    sm += 0.09;
    //Ch
    d41 = sdBox(r-vec3( 0.44+sm,0.41-sy+0.05, 0.0), vec3(0.02, 0.05, z)); //VERT
    d42 = sdBox(r-vec3( 0.44+sm+0.09,0.41-sy, 0.0), vec3(0.02, 0.10, z)); //VERT
    d43 = sdBox(r-vec3( 0.44+sm+0.05,0.41-sy, 0.0), vec3(0.05, 0.01, z)); //HORIZ //A little step
   // d43 = sdBox(r-vec3( 0.44+sm+0.04,0.41-sy, 0.0), vec3(0.06, 0.01, z)); //HORIZ //RECT connection
    d = min(d41,d42);   
    dq1 = min(dq1, d);
    dq1 = min(dq1,d43);
    //dq = min(d, d44);

    //E
    
    //T
    
    //A
    
    //Ch
        
    
       
    d = min(d1,d2);
    d = min(d, min(d3,d4));
    d = min(d, min(d5,d6));
    d = min(d, min(d7,d8));       
    d = min(d, d9);
    d = min(d, d10);
    d = min(d, d11);
    d = min(d, d12);
    d = min(d, d13);
    float dx = min(d14, d15);
    d = min(d,dx);
    dx = min(d16,d17);
    d = min(d,dx);
    float dy = min(d18,d19);
    d = min(d,dy);
    dx = min(d20,d21);
    dy = min(d22,d23);
    d = min(d,dx);
    d = min(d,dy);
   // dx = min(d24,d25);
   // d = min(d,dx);
    // = min(d,d26);
    dx = min(d24,d25);
    d = min(d,dx);
    d = min(d26,d);
    dx = min(d27,d28);
    dy = min(d,d29);
    d = min(dx,dy);
    d = min(d, dq1);
       
    return d;
}

// Normal vector - http://www.pouet.net/topic.php?which=7920&page=10 by rear
vec3 normal(vec3 p)
{
	#define dr 1e-5
	vec3 drx = vec3(dr,0,0);
	vec3 dry = vec3(0,dr,0);
	vec3 drz = vec3(0,0,dr);
	return ( vec3( dist(p+drx), dist(p+dry), dist(p+drz) ) - dist(p)) / dr;
}


//void mainImage( out vec4 fragColor, in vec2 fragCoord )
//void main(out vec4 gl_FragColor, in vec2 gl_FragCoord )
void main(void)
{       
    vec4 fragCoord = gl_FragCoord;
    vec4 fragColor = gl_FragColor;
    //vec2 uv = gl_FragCoord.xy / iResolution.xy;
	//vec2 uv = gl_FragCoord.xy / iResolution.xy;
    gl_FragColor = vec4(0.);    
    //gl_FragColor = vec4(uv, uv);   
    
        
	vec2 uv = fragCoord.xy / iResolution.xy;
    fragColor = vec4(0.);    
    vec2 r = (fragCoord.xy / iResolution.xy);
	r.x*=(iResolution.x/iResolution.y);	           
    r -=vec2(0.1, 0.8);    		    
    //r -=vec2(0.1, 0.6);  
    //vec3 camera = vec3(1.05+(sin(iTime))*cameraSpeed,cameraYz); //more to the center      	
    vec3 camera = vec3(1.75+(sin(iTime)*0.6)*cameraSpeed,cameraYz); //
    
    //vec3 ro =  vec3(r.x, r.y+1.0, -1.0);    
    //vec3 ro =  vec3(r.x+0.3, r.y+1.0 - cos(iTime)/4., -1.0 +sin(iTime)/3.);   
    vec3 ro =  vec3(r.x+0.3, r.y+1.0, -1.0 +sin(iTime)/3.);   //Slight zoom
    vec3 p = ro;  //ray origin          
	vec3 dir = normalize(p-camera); //ray direction
    float d; //distance
    
	for(int i=0; i<STEPS; i++) //Ray marching
	{
		d = dist(p);
		if(d < EPS) break;
		p = p+dir*d;
	}

    vec3 materialcolor=vec3(0.);        
    int m;  
    if (d<=cube) { m = 0; materialcolor = vec3(0.9,.9,.2);} //cube;
    else { m = 1; } //materialcolor = vec3(0.);}
            
	vec3 nor = normal(p);  // normal vector
    vec3 lightpos = vec3(1.5-sin(iTime)*5., 0.1+sin(iTime), 3.5+sin(iTime)*5.);           
    lightpos.y +=sin(iTime); // [-1., +1]
         
    vec3 lightdir = normalize(vec3(0.3,0.3,0.3)-lightpos);
   	
    float light = 1.0 + 0.01*(dot(nor,lightpos)); //intensity
    
    light *=  pow(dist(lightdir-p), 2.);
    
	//vec3 color = vec3(light);
    vec3 color = vec3(1.0-light/5.); //vec3(1.0, 1.0, 1.0);
    color = clamp( materialcolor*color, 0., 1.0);
    
    //Phong
    float dif = clamp( dot( nor, lightdir ), 0.0, 1.0 ); //iq diffuse
    vec3  ref = reflect( dir, nor );  //reflection
    float spe = pow(clamp( dot( ref, lightdir ), 0.0, 1.0 ),16.0); //specular component
        
    color+=dif/3. + spe/2.;
        
	fragColor = vec4(color, 1.0);
    fragColor.xyz = vec3(color);
    fragColor.w = m ==0 ? 1. : 0.; //
             
    vec2 pos = 2.0 * vec2(fragCoord.xy - 0.5 * iResolution.xy) / iResolution.y;
    
   //The space, stars... #28-11-2017 & electricity
   if ( fragColor.r < 0.001 && m==1)
    {     
        //To do: add more effects: specific stars, nebullas, planets, comets, black hole etc.... had meteors, but were removed;    
        vec2 n1 = N22(pos);
        float star = n1.x < 0.07 ? 0.1 : 0.;        
        star+= n1.y > 0.97 ? N1(n1.x)/1.0*(max(star, sin(iTime))) : 0.0;
        vec2 fb = fbm(pos);
        star*=max(fb.x, fb.y);       
        fragColor += star*1.4; //brighter stars
        fragColor.a = 1.0;                                 
        fragColor.xyzw+=vec4(clamp(abs(cos(iTime/3.5))*4.28, 1., 3.)*electricity(uv), 1.0); //thick line        
    }
    else //The blocks
    {
        
     const float EPSLIMIT = EPS*5.; //0.06;
        /*
     float limit = 0.0;     
     ro =  vec3(r.x, r.y+1.0, -1.0);    
     p = ro;
	 dir = normalize(p-camera);                     
	 for(int i=0; i<STEPS/3; i++)  //Second marching for the rotating block and the burst. Fewer steps and lower precision are enough.
	 {
        d = distLimitBorder(p);
		if(d < EPSLIMIT) break;
		p = p + dir * d;
	  }
        */
        
      if (d<EPSLIMIT) fragColor.xyz += electricity(uv);
            
      fragColor.w = 1.0; //Alpha
      
    }
   
    //Gamma correction
     fragColor.xyz=pow(fragColor.xyz, vec3(1.4));
     gl_FragColor = fragColor;
    
}

//void main(){} 
""");


import numpy as np

# Texture 1
im1 = io.load_crate()

# Texture with bumbs (to muliply with im1)
im2 = np.ones((20,20), 'float32')
im2[::3,::3] = 0.25 #0.5

# Texture with a plus sign (to subtract from im1)
im3 = np.zeros((30,30), 'float32')
im3[10,:] = 1.0
im3[:,10] = 1.0

# Create vetices and texture coords in two separate arrays.
# Note that combining both in one array (as in hello_quad2)
# results in better performance.
#positions = np.array([  [-0.8, -0.8, 0.0], [+0.7, -0.7, 0.0],                        [-0.7, +0.7, 0.0], [+0.8, +0.8, 0.0,] ], np.float32)
                        
positions = np.array([  [-1.0, -1.0, 0.0], [+1.0, -1.0, 0.0],                        [-1., +1., 0.0], [+1., +1., 0.0,] ], np.float32)

positions = np.array([ [-1., -1., 0.0,], [1., -1., 0.0] ,[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]  ], np.float32) 
     
#MIRROR: 
#texcoords = np.array([  [1.0, 1.0], [0.0, 1.0],
#                        [1.0, 0.0], [0.0, 0.0]], np.float32)

#mirror
texcoords = np.array([  [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], np.float32)
texcoords = np.array([  [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]], np.float32)


positions = np.array([ [-1., -1., 0.0,], [-1., 1., 0.0] ,[1.0, 1.0, 0.0], [-1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0]  ], np.float32) 


     
texcoords = np.array([  [0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, .0], [1., 1.], [1., 0.]], np.float32)
     
#only mirror horizontal
texcoords = np.array([  [0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1., 0.], [1., 1.]], np.float32)

#naodolu
#texcoords = np.array([  [0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1., 1.], [1., 0.]], np.float32)

     
#[(-1, -1), (-1, 1), (1, 1),
#                                    (-1, -1), (1, 1), (1, -1)]
                        
#TEXTURE CRATE

VERT_SHADER = """ // texture vertex shader

attribute vec3 a_position;
attribute vec2 a_texcoord;

varying vec2 v_texcoord;

void main (void) {
    // Pass tex coords
    v_texcoord = a_texcoord;
    // Calculate position
    gl_Position = vec4(a_position.x, a_position.y, a_position.z, 1.0);
}
"""

#4 Texture crates ... example  https://vispy.readthedocs.io/en/v0.2.1/examples/texturing.html
shaders.append("""

uniform sampler2D u_texture1;
uniform sampler2D u_texture2;
uniform sampler2D u_texture3;
varying vec2 v_texcoord;

/* SIMPLE MIX ... 
void main(){
    vec4 clr1 = texture2D(u_texture1, v_texcoord);
    vec4 clr2 = texture2D(u_texture2, v_texcoord);
    vec4 clr3 = texture2D(u_texture3, v_texcoord);
    
    
   
    
    
    //gl_FragColor.rgb = clr1.rgb //basic
    //gl_FragColor.rgb = clr1.rgb * clr2.r - clr3.r; //squares ... and black lines
    gl_FragColor.a = 1.0;
} 
*/

#define T texture2D(iChannel0,(s*p.zw+ceil(s*p.x))/2e2).y/(s+=s)*4
#define MAXSTEPS 202

void main(){
    vec4 clr1 = texture2D(u_texture1, v_texcoord);
    vec4 clr2 = texture2D(u_texture2, v_texcoord);
    vec4 clr3 = texture2D(u_texture3, v_texcoord);
    //vec2 iResolution = vec2(640., 360.);
    
    vec4 p=vec4(.8,0,gl_FragCoord/iResolution.y-.8);
    vec4 d = p;
    vec4 c=vec4(.6,.7,d);
    gl_FragColor=c-d.w;
    float f,s,t=2e2+sin(dot(gl_FragCoord,gl_FragCoord));
    t--;           
    for(int i=0; i<MAXSTEPS; i++) {
        p=.05*t*d,
        p.xz+=iGlobalTime,       
        s=2.,
        f=p.w+1.-T-T-T-T,
    	f<0.?gl_FragColor+=(gl_FragColor-1.-f*c.zyxw)*f*.4:gl_FragColor;
        t--;       
    }
    
    //gl_FragColor.rgb = clr1.rgb //basic
    gl_FragColor.rgb = gl_FragColor.rgb-(clr1.rgb * clr2.r - clr3.r)/3.0; //squares ... and black lines
    gl_FragColor.a = 1.0;
} 


    
""");




#Sacred computer, added 30.7.2019

'''
#https://github.com/vispy/vispy/issues/1302
Render larger than screen to texture:

size = (2048, 2048)
rendertex = gloo.Texture2D(shape=size + (4,))
fbo = gloo.FrameBuffer(rendertex, gloo.RenderBuffer(size))
And later:

with fbo:
    gloo.clear(depth=True)
    gloo.set_viewport(0, 0, *size)
    # Any drawing operation you might wish
    screenshot = gloo.read_pixels((0, 0, *size), True)
'''
# -------------------------------------------------------------------------

#canvas = Canvas(SHADERTOY)
# Input data.
#canvas.set_channel_input(noise(resolution=256, nchannels=1), i=0)

if __name__ == '__main__':
    #print("Select shader: . Landscape\n2. Hackafe\n 3. Tiny Clouds");
    
    #mRootPath = ".\\"
    mRootPath = "D:\\Py\\gl\\"
    mV = r"P:\UT\2019-07-14_16-37-18_Dramatichno_Inhuman.mp4"
    mV = r"P:\UT\2019-07-18_00-49-02_Slabo.mp4"
    mVideoPath = r"C:\Video\Vulpev 2019 18_20mbps.avi"
    mVideoPath = r"C:\Video\2019-07-18_00-45-13_Slabo.mkv"
    mVideoPathIO = r"C:\Video\Vulpev 2019 18_20mbps.avi"
    #Different than cap for the openCV if both are open, otherwise io opens black
    #mVideoPath = r"C:\Video\arnaud13_nadpisi_-1024809718.avi"
    mVideoPathIO = r"D:\Video\vlizane_28_x720_2054467656.avi_audio.avi"
    mVideoPathIO = r"C:\Video\loko-1.mp4"
    mVideoPath = r"C:\Video\loko-2.mp4"
    mVideoPathIO =  r"F:\video\CASIO\MVI_8032.AVI"
    
    #cap = cv2.VideoCapture(mVideoPath)
    cap = cv2.VideoCapture(0) #web cam #mVideoPath)
    
    mFrames = 5;
    mStart = 40 #1900;
    print(cap)
    
    frameArr = []
    
    #cap = cv2.VideoCapture(mVideoPath)
    print(cap)
    print(dir(cap))
    cap.set(cv2.CAP_PROP_POS_FRAMES, mStart) #1900)
    ret, frame = cap.read(0)
    i = 0
    #while(ret):
    while i<mFrames:
      if ret: cv2.imshow("cap", frame)     
      else: print(str(ret), str(frame))  
      cv2.waitKey(1)
      ret, frame = cap.read(0)
      i+=1
      #cv2.cvtColor()
      #cv2.cvtColor(thresh,  cv2.COLOR_BGR2GRAY)
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frameArr.append(frame)
      
    #print("Enter...")
    cap.release()
    #s = input()
    '''OK
    cap = cv2.VideoCapture(mVideoPath)
    print(cap)
    print(dir(cap))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 900)
    ret, frame = cap.read(0)
    while(ret):
      if ret: cv2.imshow("cap", frame)     
      else: print(str(ret), str(frame))  
      cv2.waitKey(5)
      ret, frame = cap.read(0)
    print("Enter...")
    cap.release()
    s = input()
    '''
    
    #"P:\UT\AGI\Coincidences_B_2019-07-21_02-42-01.mkv"
    mReadImages = 0 #1 #0
    mReadVideo = 1#0
    slow = [1,1,1,0.01,1]
    mOpencvVideo = 0 #imageio
    video = imageio.read(mVideoPathIO)
  
    for j in slow:
      print("Slow factor per index:",j)
    print("0 Hackafe\n 1. Clouds \n 2. Noise 3. Sacred Computer \n 4. Texture = 1");
    inp = input()
    i = int(inp)   
    SHADERTOY = shaders[i]
    print("Render to images? (1=TRUE, 0 - FALSE)")
    r = (int)(input())
    print(r)    
    texture = 0
    texture = 1   
    if (i ==4):
      texture = 1      
    canvas = Canvas(SHADERTOY, slow[i], r, texture)
    
    if (i ==4):      
      pass
      #canvas._timer = app.Timer(0.1, connect=canvas.on_timer, start=True)
      
    # Input data.
    canvas.set_channel_input(noise(resolution=256, nchannels=1), i=0)
    #TEXTURES iChannel! 
    
    #canvas.show()
    
    ''' NO
    ev = 0.0
    for i in range(20):
      print("on_timer PRE?")
      #canvas.on_timer(ev)
      canvas.force_timer(timeOne.time() - canvas.startTime) #?
      canvas.update()
      ev += 0.1
      #cv2.waitKey(50)    
    '''
   #cv2.waitKey(1) 
    if sys.flags.interactive == 0:
        canvas.app.run()
        
    #canvas.cap.release()
    cap.release()
