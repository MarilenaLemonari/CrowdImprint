#IMPORTS:
from enum import unique
from re import S
from turtle import mode, shape
import numpy as np
import pickle
import time
import os
from numpy.core.numerictypes import nbytes
import os.path
import random
from tqdm import tqdm
import math

#TODO: go to \examples with cd
# cd C:\PROJECTS\SocialLandmarks
# Execute .\.venv\Scripts\activate
# Go to cd C:\PROJECTS\DataDrivenInteractionFields\InteractionFieldsUMANS\examples
# Execute python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\generate_trajectories.py

#--------------------------------------------------------------------------------------------------------------
#SCENARIO SETUP:
path='C:/PROJECTS/DataDrivenInteractionFields/InteractionFieldsUMANS/examples/'
pathIF=f"{path}interactionFields/"
pathA=f"{path}agents/"
pathO=f"{path}obstacles/"
pathP=f"{path}policies/"
#IF.XML
def build_IFxml(desired_name,Ns,dictionary):
  from xml.dom import minidom
  import os 
  root = minidom.Document()
  xml = root.createElement('InteractionFields') 
  root.appendChild(xml)
  for i in range(Ns):
    for j in range(len(dictionary)):
      sourceIF=f'source{i+1}'
      sourceIF=root.createElement('InteractionField') 
      sourceIF.setAttribute('id', f'{i*10+j}')
      sourceIF.setAttribute('isVelocity', 'true')
      sourceIF.setAttribute('weight', '0')
      sourceIF.setAttribute('action_time', '0')
      sourceIF.setAttribute('inactive_time', '0')
      xml.appendChild(sourceIF)
      basis_name=dictionary[j]
      basisIF=f'matrix/{basis_name}.txt'
      basisMatrix = root.createElement('Matrix')
      basisMatrix.setAttribute('file', basisIF)
      sourceIF.appendChild(basisMatrix)
  xml_str = root.toprettyxml(indent ="\t") 
  save_path_file = desired_name 
  with open(save_path_file, "w") as f:
      f.write(xml_str)
#AGENT.XML
def build_AGENTxml(desired_name,positions,s_positions,mode,dictionary):
  from xml.dom import minidom
  import os 
  if mode == "t":
    speed='0.8'
  else:
    speed='1.0'
  root = minidom.Document()
  xml = root.createElement('Agents') 
  root.appendChild(xml)
  N=positions.shape[0]
  Ns=len(s_positions)
  for i in range(N):
    sourceIF=f'source{i+1}'
    sourceIF=root.createElement('Agent')     
    sourceIF.setAttribute('group', '0')
    sourceIF.setAttribute('max_speed', speed)
    sourceIF.setAttribute('pref_speed', speed)
    if i==0:
      sourceIF.setAttribute('max_speed', '0.0')
      sourceIF.setAttribute('pref_speed', '0.0')
    sourceIF.setAttribute('rad', '0.3')
    xml.appendChild(sourceIF)
    sourcePos = root.createElement('pos')
    sourcePos.setAttribute('x', f'{positions[i,0]}')
    sourcePos.setAttribute('y', f'{positions[i,1]}')
    sourceIF.appendChild(sourcePos)
    sourceGoal = root.createElement('goal')
    sourceGoal.setAttribute('x', '0')
    sourceGoal.setAttribute('y', '0')
    sourceIF.appendChild(sourceGoal)
    if i==0:
      sourceOr = root.createElement('orientation')
      sourceOr.setAttribute('x', '0')
      sourceOr.setAttribute('y', '1')
      sourceIF.appendChild(sourceOr)
    sourcePolicy = root.createElement('Policy')
    sourcePolicy.setAttribute('id', '0')
    sourceIF.appendChild(sourcePolicy)
    if (i in s_positions)==True:
      for j in range(len(dictionary)):
          sourcePolicy = root.createElement('InteractionField')
          sourcePolicy.setAttribute('id', f'{j}')
          sourceIF.appendChild(sourcePolicy)
  xml_str = root.toprettyxml(indent ="\t") 
  save_path_file = desired_name 
  with open(save_path_file, "w") as f:
      f.write(xml_str)
  return Ns
#SCENARIO.XML
def build_SCENARIOxml(desired_name,desired_files):
  from xml.dom import minidom
  import os 
  root = minidom.Document()
  xml = root.createElement('Simulation')
  xml.setAttribute('delta_time','0.1')
  xml.setAttribute('end_time',f"{end_time}")
  root.appendChild(xml)
  world=root.createElement('World') 
  world.setAttribute('type', 'Infinite')
  xml.appendChild(world)
  obs=root.createElement('Obstacles') 
  obs.setAttribute('file', f'{pathO}UserStudy1_obstacle.xml')
  world.appendChild(obs)
  agents=root.createElement('Agents') 
  agents.setAttribute('file', f'{pathA}Agent_{desired_files}.xml')
  xml.appendChild(agents)
  policies=root.createElement('Policies') 
  policies.setAttribute('file', f'{pathP}InteractionFieldsWithCollisionAvoidance.xml') #TODO: RVO
  xml.appendChild(policies)
  ifs=root.createElement('InteractionFields') 
  ifs.setAttribute('file', f'{pathIF}InteractionField_{desired_files}.xml')
  xml.appendChild(ifs)  
  xml_str = root.toprettyxml(indent ="\t") 
  save_path_file = desired_name 
  with open(save_path_file, "w") as f:
      f.write(xml_str)
def build_xml(n,init_positions,dictionary):
  build_IFxml(f"{pathIF}InteractionField_social.xml",1, dictionary)
  Ns=build_AGENTxml(f"{pathA}Agent_social.xml",init_positions,[0],mode,dictionary)
  build_SCENARIOxml(f"{path}Scenario_social.xml","social")

def update_gtIFxml(IFxml_file,W,actionTimes,inactiveTimes,unique_size):
  Ns=W.shape[0]
  import xml.etree.ElementTree as ET
  tree = ET.parse(IFxml_file)
  root=tree.getroot()
  for source in range(Ns):
    for basis_i in range(unique_size):
      weight=W[source,basis_i]
      element = root[source+basis_i]
      element.set('weight', str(weight))
      actionTime=actionTimes[source,basis_i]
      element.set('action_time', str(actionTime))
      inactiveTime=inactiveTimes[source,basis_i]
      element.set('inactive_time', str(inactiveTime))
      #if(actionTime_idx==basis_i):
        #element.set('action_time', str(actionTime))
      tree.write(IFxml_file, encoding = "UTF-8", xml_declaration = True)

def update_Agentxml(AgentxmlFile, or_x, or_y):
  import xml.etree.ElementTree as ET
  tree = ET.parse(AgentxmlFile)
  root=tree.getroot()
  element = root[0] #source
  element[2].set('x', f'{or_x}')
  element[2].set('y', f'{or_y}')
  tree.write(AgentxmlFile, encoding = "UTF-8", xml_declaration = True)

#--------------------------------------------------------------------------------------------------------------
#Make S_true:
def load_trajectories(file_name, resume_time):
  file = open(file_name)
  numpy_array = np.loadtxt(file, delimiter=",")
  numpy_array=numpy_array[resume_time:,:]
  return numpy_array
def make_trajectory(n,n_agents,mode,category):
  if category == "Training":
    folder = mode
  else:
    folder=f"{mode}/TestData"

  for i in range(1,n_agents):
    file=f"C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\Trajectories\{folder}\{n}_a{i}.csv"
    file2=f"C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\Trajectories\{folder}\output_{mode}{i}.csv"
    if os.path.exists(file)==True:
      os.remove(file)
    if os.path.exists(file2)==True:
      os.remove(file2)
  os.getcwd()
  os.system(f"C:\\PROJECTS\\DataDrivenInteractionFields\\InteractionFieldsUMANS\\buildConsole\\Release\\UMANS-ConsoleApplication-Windows.exe -i Scenario_social.xml -o C:\\PROJECTS\\SocialLandmarks\\SocialLandmarks_Python\\Data\\Trajectories\\{folder}")
  S_true=[]
  for i in range(1,n_agents):
    os.rename(f"C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\Trajectories\{folder}\output_{i}.csv",f"C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\Trajectories\{folder}\{n}_a{i}.csv")
    S_true.append(load_trajectories(f"C:/PROJECTS/SocialLandmarks/SocialLandmarks_Python/Data/Trajectories/{folder}/{n}_a{i}.csv", 0))
  return S_true
#---------------------------------------------------------------------------------------------------------------

def generate_instance(n,init_positions,weight,actionTimes,inactiveTimes,or_x, or_y,category,dictionary,mode):
  build_xml(n,init_positions,dictionary)
  n_agents=init_positions.shape[0]
  update_gtIFxml(f"{pathIF}InteractionField_social.xml",weight,actionTimes,inactiveTimes,len(dictionary))
  update_Agentxml(f"{pathA}Agent_social.xml",or_x,or_y)
  S_true=make_trajectory(n,n_agents,mode,category)

if __name__ ==  '__main__':

  behavior_list = ["Unidirectional_Down","Attractive_Multidirectional","Other_CircleAround", "AvoidNew", "MoveTF", "Stop"]

  dictionary = {}
  for i in range(len(behavior_list)-1):
    dictionary[i] = behavior_list[i]

  category = "Training" 
  if category == "Training":
    repeat = 1000 * len(behavior_list) # TODO:change
    prefix = 'IF_'
  elif category == "Testing":
    repeat = 100
    prefix = '_test_IF_'

  counter = 0
  mode = "SingleSwitch"

  if mode == "SingleSwitch":
    for r in tqdm(range(repeat)):
      end_time = random.randint(5,15)
      radius = end_time/2 # 5 for 10 sec

      field_1=random.randint(0,len(behavior_list)-1)
      field_2=random.randint(0,len(behavior_list)-1)

      weight=np.zeros((1,len(behavior_list)-1))
      actionTimes=np.ones((1,len(behavior_list)-1))*(-1)
      inactiveTimes=np.ones((1,len(behavior_list)-1))*(-1)
      T = random.randint(2,int(end_time-2)) # TODO: min switch

      if field_1 != 5 and field_2 != 5:
        weight[0,field_1] = 1
        weight[0,field_2] = 1

        inactiveTimes[0,field_1] = T
        actionTimes[0,field_2] = T

        inactiveTimes[0,field_2] = end_time
        actionTimes[0,field_1] = 0
      elif field_1 == 5 and field_2 != 5:
        weight[0,field_2] = 1
        inactiveTimes[0,field_2] = end_time
        actionTimes[0,field_2] = T
      elif field_1 != 5 and field_2 == 5:
        weight[0,field_1] = 1
        inactiveTimes[0,field_1] = T
        actionTimes[0,field_1] = 0

      x0 = 0
      y0 = 0
      angle = random.uniform(0, 2 * math.pi)
      x = x0 + radius * math.cos(angle)
      y = y0 + radius * math.sin(angle)
      init_positions=np.array([[x0,y0],[x,y]])


      random_angle = random.uniform(0, 2 * math.pi)
      or_x = math.cos(random_angle)
      or_y = math.sin(random_angle)

      n=str(counter)+prefix+str(field_1)+"_"+str(field_2)+"_T"+str(T)+"_d"+str(end_time)
      counter += 1

      generate_instance(n,init_positions,weight,actionTimes,inactiveTimes,or_x, or_y, category,dictionary,mode)

  else:
    print("Error: Wrong Mode. Select a valid mode.")