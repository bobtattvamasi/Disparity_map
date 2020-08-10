#!/usr/bin/python3

# Interactive rubber-band drawing on top of an image, using PySimpleGUI.
# Because PySimpleGUI doesn't (it seems) have its own double-buffering,
# I'm using PIL (Pillow), and PIL's own draw function, to create the rubber
# band. This works fine, except that PySimpleGUI isn't sending me the
# mouse up event when I have drag events enabled. So it's not really possible
# to implement a proper rubber band draw. It does sent a __TIMEOUT__ event,
# but only sometimes.

# for https://stackoverflow.com/questions/57191494/draw-rectangle-on-image-in-pysimplegui/57191725#57191725

import io
import PySimpleGUI as sg
from PIL import Image
from PIL import ImageDraw

im = Image.open("Tux.jpg").convert("RGBA") # https://upload.wikimedia.org/wikipedia/commons/5/56/Tux.jpg

def draw_frame(target,
               start=None, # Point
               end=None):  # Point
    """double-buffered redraw of the background image and the red rubber-rect"""
    buffer = im.copy()

    draw = ImageDraw.Draw(buffer)

    # optionally draw the rubber rectangle
    if start is not None and end is not None:
        draw.rectangle((start, end),
                       outline="red",
                       )

    # render our in-memory backing buffer to the visible widget
    b = io.BytesIO()
    buffer.save(b,'PNG')

    target.DrawImage(data=b.getvalue(), location=(0, 0))

    
layout = [
    [
        sg.Graph(
            canvas_size=(400, 400),
            graph_bottom_left=(0, 400),
            graph_top_right=(400, 0),
            key="graph",
            change_submits=True, # mouse click events
            drag_submits=True # mouse drag events
        ),
    ],
    [
        sg.Text("", key="info", size=(60,1))
    ]

]

window = sg.Window("draw rect on image", layout)
window.Finalize()

graph = window.Element("graph")

dragging = False
start_point, end_point = None, None

draw_frame(graph)

timeout_counter = 0

while True:
    event, values = window.Read()
    if event is None:
        break # exit

    if event == "graph":
        x,y = values["graph"]
        #print (f"mouse down at ({x},{y})")
        if not dragging:
            start_point = (x,y)
            dragging = True
        else:
            end_point = (x,y)

        draw_frame(graph, start=start_point, end=end_point)
    elif event.endswith('+UP'):

        info  = window.Element("info")
        info.Update(value=f"grabbed rectangle from {start_point} to {end_point}")

        # enable grabbing a new rect
        start_point, end_point = None, None
        dragging = False
    else:
        print("unhandled event", event, values)