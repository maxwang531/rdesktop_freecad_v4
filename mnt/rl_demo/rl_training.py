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

toolpath1 = "/usr/lib/freecad/Mod/Path/Tools/Bit/157mm_Ball_End.fctb"
toolpath2 = "/usr/lib/freecad/Mod/Path/Tools/Bit/317mm_Ball_End.fctb"


# Create save dir
save_dir = "/config/training_data/rl_model/" #save Rl-Model
checkpoint_dir = "/config/training_data/rl_model/checkpoint/"
log_dir = "/config/training_data/rl_model/log/"

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

'''Rohteil Bearbeitung'''
rohteil_simulator()
rohteil_export()
voxel()
rohteil_voxel_anzahl,rohteil_voxel_list_xyz = rohteil_voxel_lesen() #返回未转化过的体素坐标
#print(len(rohteil_voxel_anzahl)) #79872
#print("roh_voxel:",len(rohteil_voxel_list_xyz)) #79872
cutmaterial_faces,cutmaterial_edges,cutmaterial_points = cutmaterial_information()
cutmaterial_faces_rohteil = cutmaterial_faces
cutmaterial_edges_rohteil = cutmaterial_edges
cutmaterial_points_rohteil = cutmaterial_points
stl_binvox_delete()
DOC.removeObject('CutMaterial')
DOC.recompute()


#Funktion def (Zielteil)
def zielteil_export():
    Gui.Selection.addSelection(file_name_1, file_name_3)
    __objs__ = []
    __objs__.append(App.getDocument(file_name_2).getObject(file_name_3))
    Mesh.export(__objs__, u"/config/mnt/rl_demo/model_cut/Zielteil.stl")
    print("Zielteil export finished")
    del __objs__
    DOC.recompute()
def zielteil_voxel_lesen():

    filename = '/config/mnt/rl_demo/model_cut/Zielteil'
    with open(filename + '.binvox', 'rb') as f:
        zielteil_model = binvox_rw.read_as_coord_array(f)
        zielteil_voxel_anzahl = zielteil_model.data[0]
        zielteil_voxel_list = zielteil_model.data.tolist()
        zielteil_voxel_list_xyz = []
        for i in range(0, len(zielteil_voxel_list[0])):
            zielteil_voxel_list_xyz.append([zielteil_voxel_list[0][i],
                                           zielteil_voxel_list[2][i],
                                           zielteil_voxel_list[1][i]])
    return zielteil_voxel_anzahl, zielteil_voxel_list_xyz

'''Zielteil bearbeitung '''
zielteil_export()
voxel()
zielteil_voxel_anzahl,zielteil_voxel_list_xyz = zielteil_voxel_lesen()

stl_binvox_delete()
'''Rohteil-Zielteil'''
#对于zielteil以及rohteil进行差集运算
vor = rohteil_voxel_list_xyz  # voxelkoordinaten in Rohteil 这个是没有转化过的体素坐标
nach = zielteil_voxel_list_xyz  # voxelkoordinaten in Zielteil 也是未转化过的体素坐标

vor_array = np.asarray(vor)
vor_rows = vor_array.view([('', vor_array.dtype)] * vor_array.shape[1])
nach_array = np.asarray(nach)
nach_rows = nach_array.view([('', nach_array.dtype)] * nach_array.shape[1])
unterschied = np.setdiff1d(vor_rows, nach_rows).view(vor_array.dtype).reshape(-1, vor_array.shape[
    1])
unterschied_list = unterschied.tolist()  # 包含于rohteil理想切除体素坐标，未转化为标准坐标


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
            stepover = 10, stepdown = 1,
            circular_use_G2_G3_bool = 0, boundary_enforcement_bool = 0,
            name = 0):
    sur_face = PathSurface.Create('Surface%d'%(name))


    cut_pattern = ['Circular','CircularZigZag','Line','Offset','Spiral','ZigZag']
    sur_face.CutPattern = cut_pattern[cut_pattern_zahl]

    layer_mode = ['Single-pass','Multi-pass']
    sur_face.LayerMode = layer_mode[layer_mode_zahl]

    sur_face.setExpression('StepOver', None)
    sur_face.StepOver = stepover

    sur_face.setExpression('StepDown', None)
    sur_face.StepDown = stepdown

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





tool1_diameter = werkzeug(toolpath1, 'tool1') #ballend
App.getDocument(file_name_1).getObject('ToolBit001').ShapeName = "ballend"
tool2_diameter = werkzeug(toolpath2, 'tool2') #ballend
App.getDocument(file_name_1).getObject('ToolBit002').ShapeName = "ballend"

DOC.recompute()
werkzeuglist = ['tool1','tool2']
werkzeugdiameter = [tool1_diameter,tool2_diameter]


#ENV Aufbau
class MyEnv(Env):
    def __init__(self):


        #def surface(werkzeugname,boundaryadjustment, cut_pattern_zahl=2,stepover=50, stepdown=5
        self.cutmaterial_faces_rohteil = cutmaterial_faces_rohteil
        self.cutmaterial_edges_rohteil = cutmaterial_edges_rohteil
        self.cutmaterial_points_rohteil = cutmaterial_points_rohteil
        self.roh_voxel_anzahl = len(rohteil_voxel_list_xyz)
        self.ziel_voxel_anzahl = len(zielteil_voxel_list_xyz)
        self.roh_ziel_voxel_anzahl = len(unterschied_list)
        self.werkzeuglist = werkzeuglist
        self.werkzeugdiameter = werkzeugdiameter
        self.cutpattern = ['Circular','CircularZigZag','Line','Offset','Spiral','ZigZag']
        self.stepover = [10, 15, 20, 25 , 30, 35, 40, 45, 50,
                         55, 60, 65, 70, 75, 80, 85, 90, 95,100]

        self.action_space = spaces.MultiDiscrete([len(self.werkzeuglist),len(self.cutpattern),len(self.stepover)])
        low = [0,0,0,0,0,0]
        high = [self.roh_voxel_anzahl,self.roh_voxel_anzahl,self.roh_voxel_anzahl,1000000,1000000,1000000]
        low_np = np.array(low,dtype=np.float32)
        high_np = np.array(high,dtype=np.float32)
        self.observation_space = spaces.Box(low_np,high_np,dtype=np.float32)
        self.voxel_state = None
        self.time_length = 64

    def step(self, action):
        surface(self.werkzeuglist[list(action)[0].item()],self.werkzeugdiameter[list(action)[0].item()],
                cut_pattern_zahl= list(action)[1].item(),stepover= self.stepover[list(action)[2].item()],
                stepdown=5)
        simulator()
        export()
        voxel()
        voxel_anzahl, voxel_list_xyz = voxel_lesen()
        cutmaterial_faces, cutmaterial_edges, cutmaterial_points = cutmaterial_information()
        stl_binvox_delete()
        delete_cutmaterial()
        delete_surface()

        #Reward Berechnung
        #Step1 加工切除的体素，未转化为标准坐标
        vor_1 = rohteil_voxel_list_xyz  # voxelkoordinaten in Rohteil 这个是没有转化过的体素坐标
        nach_1 = voxel_list_xyz  # voxelkoordinaten nach der Bearbeitung 也是未转化过的体素坐标

        vor_1_array = np.asarray(vor_1)
        vor_1_rows = vor_1_array.view([('', vor_1_array.dtype)] * vor_1_array.shape[1])
        nach_1_array = np.asarray(nach_1)
        nach_1_rows = nach_1_array.view([('', nach_1_array.dtype)] * nach_1_array.shape[1])
        unterschied_1 = np.setdiff1d(vor_1_rows, nach_1_rows).view(vor_1_array.dtype).reshape(-1, vor_1_array.shape[
            1])  # https://blog.csdn.net/weixin_40264653/article/details/105540049
        unterschied_1_list = unterschied_1.tolist()  # 加工切除的体素，未转化为标准坐标
        print("cutVoxel_1:",len(unterschied_1_list)) #加工切除的体素
        reward1 = 1*len(unterschied_1_list)
        # Step2 加工后模型与zielteil的差距,加工区域内未切除体素
        vor_2 = voxel_list_xyz  # voxelkoordinaten nach der Bearbeitung 也是未转化过的体素坐标
        nach_2 = zielteil_voxel_list_xyz  # voxelkoordinaten in zielteil 这个是没有转化过的体素坐标

        vor_2_array = np.asarray(vor_2)
        vor_2_rows = vor_2_array.view([('', vor_2_array.dtype)] * vor_2_array.shape[1])
        nach_2_array = np.asarray(nach_2)
        nach_2_rows = nach_2_array.view([('', nach_2_array.dtype)] * nach_2_array.shape[1])
        unterschied_2 = np.setdiff1d(vor_2_rows, nach_2_rows).view(vor_2_array.dtype).reshape(-1, vor_2_array.shape[
            1])  # https://blog.csdn.net/weixin_40264653/article/details/105540049
        unterschied_2_list = unterschied_2.tolist()  # 加工切除的体素，未转化为标准坐标
        print("cutVoxel_2:", len(unterschied_2_list))  # 加工后模型与zielteil的差距
        reward2 = -1*len(unterschied_2_list)

        reward = reward1+reward2
        print("total_reward:",reward)


        self.time_length -= 1
        if self.time_length <= 0:
            done = True
        else:
            done = False
        info = {'roh_voxel':self.roh_voxel_anzahl,'ziel_voxel':self.ziel_voxel_anzahl
                ,'model_voxel':len(voxel_list_xyz),'cut_voxel':len(unterschied_1_list),'not_cut_voxel':len(unterschied_2_list)}
        # observation
        observation_information = [len(voxel_list_xyz), len(unterschied_1_list), len(unterschied_2_list), cutmaterial_faces,cutmaterial_edges,cutmaterial_points]
        self.voxel_state = observation_information
        print("_________________________________________________________________")
        return np.array(self.voxel_state, dtype=np.float32), reward, done, info

    def render(self):
        pass
    def reset(self):
        self.voxel_state = [self.roh_voxel_anzahl,0,self.roh_ziel_voxel_anzahl,self.cutmaterial_faces_rohteil,self.cutmaterial_edges_rohteil,self.cutmaterial_points_rohteil]
        self.time_length = 64
        return np.array(self.voxel_state, dtype=np.float32)
    def close(self):
        env.close()

env = MyEnv()

#Callback Funktion
class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


callback = TrainAndLoggingCallback(check_freq=128, save_path=checkpoint_dir)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, gamma=0 ,n_steps = 64)
model.learn(total_timesteps=500, callback=callback)


# Create save dir
os.makedirs(save_dir, exist_ok=True)

# The model will be saved under PPO2_tutorial.zip
model.save(save_dir + "/demo1_500")

App.closeDocument(file_name_2)
Gui.doCommand('exit()')

