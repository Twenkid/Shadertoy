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

"""
Shadertoy demo. You can copy-paste shader code from an example on
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
uniform vec4      iMouse;                // mouse pixel coords
uniform vec4      iDate;                 // (year, month, day, time in seconds)
uniform float     iSampleRate;           // sound sample rate (i.e., 44100)
uniform sampler2D iChannel0;             // input channel. XX = 2D/Cube
uniform sampler2D iChannel1;             // input channel. XX = 2D/Cube
uniform sampler2D iChannel2;             // input channel. XX = 2D/Cube
uniform sampler2D iChannel3;             // input channel. XX = 2D/Cube
uniform vec3      iChannelResolution[4]; // channel resolution (in pixels)
uniform float     iChannelTime[4];       // channel playback time (in sec)
//uniform float     iTime;

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


class Canvas(app.Canvas):

    def __init__(self, shadertoy=None):
        app.Canvas.__init__(self, keys='interactive')
        if shadertoy is None:
            shadertoy = """
            void main(void)
            {
                vec2 uv = gl_FragCoord.xy / iResolution.xy;
                gl_FragColor = vec4(uv,0.5+0.5*sin(iGlobalTime),1.0);
            }"""
        self.program = gloo.Program(vertex, fragment % shadertoy)

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

    def on_timer(self, event):
       #self.program['iTime'] = event.elapsed
        self.program['iGlobalTime'] = event.elapsed
        self.program['iDate'] = get_idate()  # used in some shadertoy exs
        self.update()
        self.frame+=1  #24-11-2018
        self.frameTime.append((self.frame, event.elapsed))
        if (self.frame%frame_divider ==0):
          ln = len(self.frameTime);
          fr1,t1 = self.frameTime[ln-1]
          fr2,t2 = self.frameTime[ln-2]
          fps = (fr1-fr2)/(t1-t2)
          #print({:04.2f} fps, end=", ");
          print(" %0.2f"% fps, end=", ");
          sys.stdout.flush()

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



# -------------------------------------------------------------------------

#canvas = Canvas(SHADERTOY)
# Input data.
#canvas.set_channel_input(noise(resolution=256, nchannels=1), i=0)

if __name__ == '__main__':
    #print("Select shader: . Landscape\n2. Hackafe\n 3. Tiny Clouds");
    print("0 Hackafe\n 1. Clouds \n");
    inp = input()
    i = int(inp)
    print(i)
    SHADERTOY = shaders[i]
    
    canvas = Canvas(SHADERTOY)
    # Input data.
    canvas.set_channel_input(noise(resolution=256, nchannels=1), i=0)
    
    canvas.show()
    if sys.flags.interactive == 0:
        canvas.app.run()
