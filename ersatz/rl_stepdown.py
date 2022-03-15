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

#stable-Baselines
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack , SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import  check_env



#fileinformation
filepath = '/config/eingabe_model/model.FCStd'
file_name_1 = 'model'
file_name_2 = "model"
file_name_3 = 'Solid'
operation_file = '/config/ausgabe_ngc/operation_parameter.csv'

toolpath1 =  "/usr/lib/freecad/Mod/Path/Tools/Bit/317mm_Endmill.fctb"
toolpath2 =  "/usr/lib/freecad/Mod/Path/Tools/Bit/317mm_Endmill.fctb"


# Create save dir
save_dir = "/config/training_data/rl_model/" #save Rl-Model
checkpoint_dir = "/config/training_data/rl_model/checkpoint/"
log_dir = "/config/training_data/rl_model/log/"

#Model Use
csv_file = '/config/ausgabe_ngc/operation_parameter.csv'
gcodePath_surface = '/config/ausgabe_ngc/txt/surface_operation_stepdown.txt'
gcodePath_surface_neu = '/config/ausgabe_ngc/neu_txt/surface_operation_stepdown.txt'



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

#stl/binvox delete
def stl_binvox_delete(): #delete binvox and stl
    os.system("sh /config/mnt/rl_demo/model_cut/delete.sh")
#voxelisierung
def voxel():
    os.system("sh /config/mnt/rl_demo/model_cut/model.sh")

#Cutmaterial Information
def cutmaterial_information(): #读取一些模拟后模型的数据
    cut = App.ActiveDocument.CutMaterial
    cutmaterial_faces = cut.Mesh.CountFacets
    cutmaterial_edges = cut.Mesh.CountEdges
    cutmaterial_points = cut.Mesh.CountPoints
    return cutmaterial_faces,cutmaterial_edges,cutmaterial_points

#Funktion def (Rohteil)
def rohteil_simulator():
    Gui.runCommand('Path_Simulator', 0)
    a.pathSimulation.SetupSimulation()  # Simulation Reset
    a.pathSimulation.SimFF()
    print("Rohteilsimulator is finished")
    a.pathSimulation.accept()
    DOC.recompute()
    Gui.Control.closeDialog()
def rohteil_export():
    Gui.Selection.addSelection(file_name_1, 'CutMaterial')
    __objs__ = []
    __objs__.append(App.getDocument(file_name_2).getObject("CutMaterial"))
    import Mesh
    Mesh.export(__objs__, u"/config/mnt/rl_demo/model_cut/Rohteil.stl")
    print("rohteil export finished")
    del __objs__
    DOC.recompute()
def rohteil_voxel_lesen(): # for rohteil 40x40x40
    filename = '/config/mnt/rl_demo/model_cut/Rohteil'
    with open(filename + '.binvox', 'rb') as f:
        rohteil_model = binvox_rw.read_as_coord_array(f)
        rohteil_voxel_anzahl = rohteil_model.data[0] #rohteil中体素个数
        rohteil_voxel_list = rohteil_model.data.tolist()
        rohteil_voxel_list_xyz = []
        for i in range(0,len(rohteil_voxel_list[0])):
            rohteil_voxel_list_xyz.append([rohteil_voxel_list[0][i],
                                           rohteil_voxel_list[2][i],
                                           rohteil_voxel_list[1][i]])
    return rohteil_voxel_anzahl,rohteil_voxel_list_xyz

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
#3D Surface
def surface(werkzeugname,
            boundaryadjustment ,cut_pattern_zahl = 2, layer_mode_zahl = 0,
            stepover = 10, depthoffset = 0,
            circular_use_G2_G3_bool = 0, boundary_enforcement_bool = 0,
            name = 0):
    sur_face = PathSurface.Create('Surface%d'%(name))


    cut_pattern = ['Circular','CircularZigZag','Line','Offset','Spiral','ZigZag']
    sur_face.CutPattern = cut_pattern[cut_pattern_zahl]

    layer_mode = ['Single-pass','Multi-pass']
    sur_face.LayerMode = layer_mode[layer_mode_zahl]

    sur_face.setExpression('StepOver', None)
    sur_face.StepOver = stepover

    sur_face.setExpression('DepthOffset', None)
    sur_face.DepthOffset = depthoffset

    circular_use_G2_G3 = ['true', '']
    sur_face.CircularUseG2G3 = bool(circular_use_G2_G3[circular_use_G2_G3_bool])

    boundary_enforcement = ['true','']
    sur_face.BoundaryEnforcement = bool(boundary_enforcement[boundary_enforcement_bool])

    sur_face.setExpression('BoundaryAdjustment', None)
    sur_face.BoundaryAdjustment = boundaryadjustment

    Gui.Selection.addSelection(file_name_1, 'Surface%d' % (name))
    App.getDocument(file_name_1).getObject('Surface%d' % (name)).ToolController = App.getDocument(
        file_name_1).getObject(werkzeugname)

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

tool1_diameter = werkzeug(toolpath1, 'tool1')
App.getDocument(file_name_1).getObject('ToolBit001').ShapeName = "endmill"
tool2_diameter = werkzeug(toolpath2, 'tool2')
App.getDocument(file_name_1).getObject('ToolBit002').ShapeName = "endmill"

DOC.recompute()
werkzeuglist = ['tool1','tool2']
werkzeugdiameter = [tool1_diameter,tool2_diameter]


operation = pd.read_csv(operation_file)
auswahl_werkzeug = operation.iloc[0,1]
auswahl_werkzeug_diamenter = operation.iloc[1,1]
auswhal_cutpattern = int(operation.iloc[2,1])
auswahl_stepover = int(operation.iloc[3,1])


#werkzeugweg
depth = [10,9,8,7,6,5,4,3,2,1,0]
for i in range (0,11):
    surface(auswahl_werkzeug,auswahl_werkzeug_diamenter,cut_pattern_zahl=auswhal_cutpattern,stepover=auswahl_stepover,depthoffset=depth[i],name = i)
print("--- done ---")

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
print("--- export finished ---")


data_surface = []
for line in open('/config/ausgabe_ngc/txt/surface_operation_stepdown.txt',"r"):
    data_surface.append(line)
if len(data_surface) > 12:
    data_surface[12] = 'M3 S2000'
data_surface.insert(13, 'G0 A90\n')
data_surface.insert(14, 'F2000\n')
f=open('/config/ausgabe_ngc/neu_txt/surface_operation_neu_stepdown.txt',"w")
for line in data_surface:
    f.write(line+'\n')
f.close()
f=open('/config/ausgabe_ngc/surface_operation_neu_stepdown.ngc',"w")
for line in data_surface:
    f.write(line+'\n')
f.close()

App.closeDocument(file_name_1)
Gui.doCommand('exit()')
