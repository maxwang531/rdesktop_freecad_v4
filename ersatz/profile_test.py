import sys

# Env import
import numpy
from gym import Env
from gym import spaces, core, utils ,logger
from gym.utils import seeding


import numpy as np
import pandas as pd
import random
import math
from math import pi, sin, cos
import csv
from time import sleep

#Sys Path
BINVOXPATH="/config/mnt/rl_demo/utils"
#Sys import
import os
import sys
sys.path.append(BINVOXPATH)

import binvox_rw
import Mesh

#FreeCAD import
import FreeCAD as App
import FreeCADGui as Gui
import setuptools

from FreeCAD import Base, Rotation, Vector
import Part
import Path
import Draft
from PathScripts import PathJob
from PathScripts import PathJobGui

from PathScripts import PathProfile
from PathScripts import PathAdaptive
from PathScripts import PathPocket
from PathScripts import PathSurface

import PathScripts.PathDressupDogbone as PathDressupDogbone
import PathScripts.PathDressupHoldingTags as PathDressupHoldingTags

from PathScripts import PathGeom
from PathScripts import PathPostProcessor
from PathScripts import PathUtil
from PathScripts import PathSimulatorGui as a

from PathScripts import PathToolBit
from PathScripts import PathToolController


#fileinformation
filepath = '/config/eingabe_model/model.FCStd'
file_name_1 = 'model'
file_name_2 = "model"
file_name_3 = 'Solid'

toolpath1 = "/usr/lib/freecad/Mod/Path/Tools/Bit/317mm_long_Ball_End.fctb"
toolpath2 = "/usr/lib/freecad/Mod/Path/Tools/Bit/317mm_Ball_End.fctb"


#Model Use
csv_file = '/config/ausgabe_ngc/operation_parameter.csv'
gcodePath_surface = '/config/ausgabe_ngc/txt/profile_operation.txt'
gcodePath_surface_neu = '/config/ausgabe_ngc/neu_txt/profile_operation.txt'



#Datei öffnen
DOC=App.openDocument(filepath)
DOC.recompute()
DOC = App.activeDocument()
DOC.recompute()

#Model = Part1
Part1 = DOC.getObject(file_name_3)

#Job creat
Gui.activateWorkbench("PathWorkbench") #Path Workbrench activate
job = PathJob.Create('Job', [Part1], None)
job.ViewObject.Proxy = PathJobGui.ViewProvider(job.ViewObject)

#operation about Stock
stock = job.Stock
stock.setExpression('ExtXneg',None)
stock.ExtXneg = 0.00
stock.setExpression('ExtXpos',None)
stock.ExtXpos = 0.00
stock.setExpression('ExtYneg',None)
stock.ExtYneg = 0.00
stock.setExpression('ExtYpos',None)
stock.ExtYpos = 0.00
stock.setExpression('ExtZneg',None)
stock.ExtZneg = 0.00
stock.setExpression('ExtZpos',None)
stock.ExtZpos = 0.00

#加工操作
#刀具
def werkzeug(toolpath,name2,horizrapid = "15mm/s",vertrapid = "2mm/s",):
    name1 = PathToolBit.Declaration(toolpath)

    tool = PathToolController.Create(name2)
    tool.setExpression('HorizRapid', None)
    tool.HorizRapid = horizrapid
    tool.setExpression('VertRapid', None)
    tool.VertRapid = vertrapid

    name3 = tool.Tool
    name3.Label = name1['name']
    name3.BitShape = name1['shape']
    name3.CuttingEdgeHeight = name1['parameter']['CuttingEdgeHeight']
    name3.Diameter = name1['parameter']['Diameter']
    name3.Length = name1['parameter']['Length']
    name3.ShankDiameter = name1['parameter']['ShankDiameter']
    name3.recompute()
    name3.ViewObject.Visibility = True
    name3.recompute()
    return name3.Diameter
#profile
def profile(werkzeugname, name = 0, stepdown = 1):
    profile = PathProfile.Create('Profile%d'%(name))

    profile.setExpression('StepDown',None)
    profile.StepDown = stepdown

    Gui.Selection.addSelection(file_name_1, 'Profile%d'%(name))
    App.getDocument(file_name_1).getObject('Profile%d'%(name)).ToolController = App.getDocument(
        file_name_1).getObject(werkzeugname)
    profile.recompute()
    DOC.recompute()
def simulator():
    Gui.runCommand('Path_Simulator', 0)
    a.pathSimulation.SetupSimulation()  # Simulation Reset
    a.pathSimulation.SimFF()
    while a.pathSimulation.iprogress < a.pathSimulation.numCommands:
        Gui.updateGui()
        sleep(0.001)
    print("Tasks are finished")
    a.pathSimulation.accept()
    DOC.recompute()
    Gui.Control.closeDialog()
def export():
    Gui.Selection.addSelection(file_name_1, 'CutMaterial')
    __objs__ = []
    __objs__.append(App.getDocument(file_name_2).getObject("CutMaterial"))

    Mesh.export(__objs__, u"/config/mnt/rl_demo/model_cut/Curved-CutMaterial.stl")
    print("export finished")
    del __objs__
    DOC.recompute()
def delete_cutmaterial():
    DOC.removeObject('CutMaterial')
    DOC.recompute()
def delete_surface():
    DOC.removeObject('Surface0')
    DOC.recompute()
def voxel_lesen():
    import binvox_rw
    filename = '/config/mnt/rl_demo/model_cut/Curved-CutMaterial'
    with open(filename + '.binvox', 'rb') as f:
        model = binvox_rw.read_as_coord_array(f)
        voxel_anzahl = model.data[0]
        voxel_list = model.data.tolist()
        voxel_list_xyz = []
        for i in range(0, len(voxel_list[0])):
            voxel_list_xyz.append([voxel_list[0][i],
                                    voxel_list[2][i],
                                    voxel_list[1][i]])
    return voxel_anzahl, voxel_list_xyz





tool1_diameter = werkzeug(toolpath1, 'tool1') #ballend
App.getDocument(file_name_1).getObject('ToolBit001').ShapeName = "ballend"
tool2_diameter = werkzeug(toolpath2, 'tool2') #ballend
App.getDocument(file_name_1).getObject('ToolBit002').ShapeName = "ballend"

DOC.recompute()
werkzeuglist = ['tool1','tool2']
werkzeugdiameter = [tool1_diameter,tool2_diameter]
auswahl_werkzeug = werkzeuglist[1]

profile(auswahl_werkzeug)
job.PostProcessorOutputFile = gcodePath_surface
job.PostProcessor = 'linuxcnc'
job.PostProcessorArgs = '--no-show-editor'
postlist = []
currTool = None
for obj in job.Operations.Group:
    print( obj.Name)
    tc = PathUtil.toolControllerForOp(obj)
    if tc is not None:
        if tc.ToolNumber != currTool:
            postlist.append(tc)
            currTool = tc.ToolNumber
    postlist.append(obj)

post = PathPostProcessor.PostProcessor.load(job.PostProcessor)
gcode = post.export(postlist, gcodePath_surface, job.PostProcessorArgs)
DOC.recompute()
print("--- done ---")

data_surface = []
for line in open('/config/ausgabe_ngc/txt/profile_operation.txt',"r"):
    data_surface.append(line)
if len(data_surface) > 12:
    data_surface[12] = 'M3 S2000'
data_surface.insert(13, 'G0 A90\n')
data_surface.insert(14, 'F2000\n')
f=open('/config/ausgabe_ngc/neu_txt/profile_operation_neu.txt',"w")
for line in data_surface:
    f.write(line+'\n')
f.close()
f=open('/config/ausgabe_ngc/profile_operation_neu.ngc',"w")
for line in data_surface:
    f.write(line+'\n')
f.close()


App.closeDocument(file_name_1)
Gui.doCommand('exit()')
