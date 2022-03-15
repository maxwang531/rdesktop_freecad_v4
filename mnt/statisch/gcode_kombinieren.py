import os

file_top = '/config/ausgabe_ngc/txt/top_operation.txt'
file_site1 = '/config/ausgabe_ngc/txt/site1_operation.txt'
file_site2 = '/config/ausgabe_ngc/txt/site2_operation.txt'
file_site3 = '/config/ausgabe_ngc/txt/site3_operation.txt'
file_site4 = '/config/ausgabe_ngc/txt/site4_operation.txt'
file_top_neu = '/config/ausgabe_ngc/neu_txt/top_operation.txt'
file_site1_neu = '/config/ausgabe_ngc/neu_txt/site1_operation.txt'
file_site2_neu = '/config/ausgabe_ngc/neu_txt/site2_operation.txt'
file_site3_neu = '/config/ausgabe_ngc/neu_txt/site3_operation.txt'
file_site4_neu = '/config/ausgabe_ngc/neu_txt/site4_operation.txt'


data_top = []
for line in open(file_top,"r"):
    data_top.append(line)
if len(data_top) > 12:
    data_top[12] = 'M3 S2000'
data_top.insert(13, 'G0 A90\n')
data_top.insert(14, 'F2000\n')
f=open('/config/ausgabe_ngc/neu_txt/top_operation_neu.txt',"w")
for line in data_top:
    f.write(line+'\n')
f.close()

data_site1 = []
for line in open('/config/ausgabe_ngc/txt/site1_operation.txt',"r"):
    data_site1.append(line)
if len(data_site1) > 12:
    data_site1[12] = 'M3 S2000'
data_site1.insert(13, 'G0 A0\n')
data_site1.insert(14, 'G0 B180\n')
data_site1.insert(15, 'F2000\n')
f=open('/config/ausgabe_ngc/neu_txt/site1_operation_neu.txt',"w")
for line in data_site1:
    f.write(line+'\n')
f.close()

data_site2 = []
for line in open('/config/ausgabe_ngc/txt/site2_operation.txt',"r"):
    data_site2.append(line)
if len(data_site2) > 12:
    data_site2[12] = 'M3 S2000'
data_site2.insert(13, 'G0 B270\n')
data_site2.insert(14, 'F2000\n')
f=open('/config/ausgabe_ngc/neu_txt/site2_operation_neu.txt',"w")
for line in data_site2:
    f.write(line+'\n')
f.close()

data_site3 = []
for line in open('/config/ausgabe_ngc/txt/site3_operation.txt',"r"):
    data_site3.append(line)
if len(data_site3) > 12:
    data_site3[12] = 'M3 S2000'
data_site3.insert(13, 'G0 B0\n')
data_site3.insert(14, 'F2000\n')
f=open('/config/ausgabe_ngc/neu_txt/site3_operation_neu.txt',"w")
for line in data_site3:
    f.write(line+'\n')
f.close()

data_site4 = []
for line in open('/config/ausgabe_ngc/txt/site4_operation.txt',"r"):
    data_site4.append(line)
if len(data_site4) > 12:
    data_site4[12] = 'M3 S2000'
data_site4.insert(13, 'G0 B90\n')
data_site4.insert(14, 'F2000\n')
f=open('/config/ausgabe_ngc/neu_txt/site4_operation_neu.txt',"w")
for line in data_site4:
    f.write(line+'\n')
f.close()

data=data_top+data_site1+data_site2+data_site3+data_site4
f=open('/config/ausgabe_ngc/neu_txt/operation_neu.txt',"w")
for line in data:
    f.write(line+'\n')
f.close()

f=open('/config/ausgabe_ngc/statisch_operation.ngc',"w")
for line in data:
    f.write(line+'\n')
f.close()

Gui.doCommand('exit()')

Gui.doCommand('exit()')
