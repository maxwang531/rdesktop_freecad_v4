#FreeCAD import
import os
import csv
import pandas as pd
import FreeCAD as App
import FreeCADGui as Gui
from FreeCAD import Base, Rotation, Vector

import Part
import Draft
import Mesh

path = "/config/eingabe_model"
datanames = os.listdir(path)
for i in datanames:
    if os.path.splitext(i)[1]=='.stp':
        eingabe_model="/config/eingabe_model/"+i
        


#eingabe_model="/config/eingabe_model/val2.stp"

FreeCAD.loadFile(eingabe_model)
App.getDocument("Unnamed").saveAs(u"/config/eingabe_model/model.FCStd")

filepath_original = '/config/eingabe_model/model.FCStd'

filename = 'Unnamed'

DOC=App.openDocument(filepath_original)
DOC.recompute()
DOC = App.activeDocument()
DOC.recompute()
obj = App.ActiveDocument.getObject('Solid')
xmin = obj.Shape.BoundBox.XMin
ymin = obj.Shape.BoundBox.YMin
zmin = obj.Shape.BoundBox.ZMin
xmax = obj.Shape.BoundBox.XMax
ymax = obj.Shape.BoundBox.YMax
zmax = obj.Shape.BoundBox.ZMax

Modellange= xmax-xmin
list = [Modellange]
test = pd.DataFrame(data=list)
modellangefile = '/config/eingabe_model/csv_file/modellange.csv'
test.to_csv(modellangefile)


box0 = DOC.addObject('Part::Box', 'Box')
box0.Width = Modellange
box0.Length = Modellange
box0.Height = Modellange



Vec = Base.Vector
pos=Vec(xmin,ymin,zmin)
rot=FreeCAD.Rotation(Vec(0,0,1),0)
center=Vec(0,0,0)
box0.Placement=FreeCAD.Placement(pos,rot,center)

cut = DOC.addObject('Part::Common', 'Common')
cut.Base = obj
cut.Tool = box0

DOC.recompute()

__objs__=[]
__objs__.append(App.getDocument(filename).getObject("Common"))
Mesh.export(__objs__,u"/config/eingabe_model/bool_model/Common.stl")
del __objs__
DOC.recompute()

App.getDocument(filename).saveAs(u"/config/eingabe_model/bool_model/Common.FCStd")
DOC.recompute()

os.system("sh /config/eingabe_model/bool_model/model_ssdnet.sh")

Gui.doCommand('exit()')
