#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vispy: gallery 2, testskip
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# Extened by Todor Arnaudov, 25-11-2018
# -- Current shadertoy variables do not match, need adjustment
#... added Tiny Clouds webGL 1.0 port
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
# -------------------------------------------------------------------------

canvas = Canvas(SHADERTOY)
# Input data.
canvas.set_channel_input(noise(resolution=256, nchannels=1), i=0)

if __name__ == '__main__':

    canvas.show()
    if sys.flags.interactive == 0:
        canvas.app.run()
