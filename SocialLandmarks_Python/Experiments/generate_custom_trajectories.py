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
  # Placeholder:
  i=0
  sourceIF=f'source{i+1}'
  sourceIF=root.createElement('Agent')   
  sourceIF.setAttribute('max_speed', '0')
  sourceIF.setAttribute('pref_speed', '0')
  sourceIF.setAttribute('rad', '0.0')
  xml.appendChild(sourceIF)
  sourcePos = root.createElement('pos')
  sourcePos.setAttribute('x', '1')
  sourcePos.setAttribute('y', '1')
  sourceIF.appendChild(sourcePos)
  sourceGoal = root.createElement('goal')
  sourceGoal.setAttribute('x', '0')
  sourceGoal.setAttribute('y', '0')
  sourceIF.appendChild(sourceGoal)
  sourcePolicy = root.createElement('Policy')
  sourcePolicy.setAttribute('id', '0')
  sourceIF.appendChild(sourcePolicy)
  # Rest:
  for i in range(N):
    sourceIF=f'source{i+2}'
    sourceIF=root.createElement('Agent')     
    sourceIF.setAttribute('group', '0')
    sourceIF.setAttribute('max_speed', speed)
    sourceIF.setAttribute('pref_speed', speed)
    if i % 2 == 0:
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
    if i % 2 == 0:
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
          if i == 0:
            sourcePolicy.setAttribute('id', f'{j}')
          else:
            m = int(i / 2)
            sourcePolicy.setAttribute('id', f'{m}{j}')
          sourceIF.appendChild(sourcePolicy)
  xml_str = root.toprettyxml(indent ="\t") 
  save_path_file = desired_name 
  with open(save_path_file, "w") as f:
      f.write(xml_str)
  return Ns
#SCENARIO.XML
def build_SCENARIOxml(desired_name,desired_files, end_time):
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
def build_xml(init_positions,source_list, dictionary, end_time):
  build_IFxml(f"{pathIF}InteractionField_social.xml",len(source_list), dictionary)
  Ns=build_AGENTxml(f"{pathA}Agent_social.xml",init_positions,source_list,mode,dictionary)
  build_SCENARIOxml(f"{path}Scenario_social.xml","social",end_time)

def update_gtIFxml(IFxml_file,W,actionTimes,inactiveTimes,unique_size, groupID):
  Ns=W.shape[0]
  import xml.etree.ElementTree as ET
  tree = ET.parse(IFxml_file)
  root=tree.getroot()
  # for source in range(Ns):
  source = unique_size * groupID
  for basis_i in range(unique_size):
    weight=W[0,basis_i]
    element = root[source+basis_i]
    element.set('weight', str(weight))
    actionTime=actionTimes[0,basis_i]
    element.set('action_time', str(actionTime))
    inactiveTime=inactiveTimes[0,basis_i]
    element.set('inactive_time', str(inactiveTime))
    #if(actionTime_idx==basis_i):
      #element.set('action_time', str(actionTime))
    element.set('group', str(groupID))
    tree.write(IFxml_file, encoding = "UTF-8", xml_declaration = True)

def update_Agentxml(AgentxmlFile, or_x, or_y, groupID, init_positions):
  start_agent = groupID * 2 + 1
  import xml.etree.ElementTree as ET
  tree = ET.parse(AgentxmlFile)
  root=tree.getroot()
  element = root[start_agent] #source
  element.set('group', str(groupID))
  element[2].set('x', f'{or_x}')
  element[2].set('y', f'{or_y}')

  element1 = root[start_agent+1] #1st agent
  element1.set('group', str(groupID)) 
  pos_x = init_positions[1,0]
  pos_y = init_positions[1,1]
  element1[0].set('x', f'{pos_x}')
  element1[0].set('y', f'{pos_y}')
  
  tree.write(AgentxmlFile, encoding = "UTF-8", xml_declaration = True)

#--------------------------------------------------------------------------------------------------------------
#Make S_true:
def load_trajectories(file_name, resume_time):
  file = open(file_name)
  numpy_array = np.loadtxt(file, delimiter=",")
  numpy_array=numpy_array[resume_time:,:]
  return numpy_array
def make_trajectory(n_agents,mode):
  # if category == "Training":
  #   folder = mode
  # else:
  #   folder=f"{mode}/TestData"

  folder  = mode

  for i in range(1,n_agents+1):
    file=f"C:\\PROJECTS\\SocialLandmarks\\SocialLandmarks_Python\\Data\\Trajectories\\{folder}\\agent_{i}.csv"
    file2=f"C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\Trajectories\{folder}\output_{mode}{i}.csv"
    if os.path.exists(file)==True:
      os.remove(file)
    if os.path.exists(file2)==True:
      os.remove(file2)
  os.getcwd()
  os.system(f"C:\\PROJECTS\\DataDrivenInteractionFields\\InteractionFieldsUMANS\\buildConsole\\Release\\UMANS-ConsoleApplication-Windows.exe -i Scenario_social.xml -o C:\\PROJECTS\\SocialLandmarks\\SocialLandmarks_Python\\Data\\Trajectories\\{folder}")
  S_true=[]
  index_list = list(np.arange(0,n_agents+1)*2)[1:]
  #index_list = [1] + index_list
  counter  = 1
  for i in range(n_agents*2+1):
    if i in index_list:
      os.rename(f"C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\Trajectories\{folder}\output_{i}.csv",f"C:\\PROJECTS\\SocialLandmarks\\SocialLandmarks_Python\\Data\\Trajectories\\{folder}\\agent_{counter}.csv")
      S_true.append(load_trajectories(f"C:/PROJECTS/SocialLandmarks/SocialLandmarks_Python/Data/Trajectories/{folder}/agent_{counter}.csv", 0))
      counter += 1
    else: 
      os.remove(f"C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\Trajectories\{folder}\output_{i}.csv")
  return S_true
#---------------------------------------------------------------------------------------------------------------

def generate_instance(init_positions,weight,actionTimes,inactiveTimes,or_x, or_y,dictionary, groupID):
  # build_xml(init_positions,dictionary, end_time)
  # start_agent = groupID * 2
  n_agents=init_positions.shape[0]
  fieldGroup = groupID
  agentGroup = groupID
  update_gtIFxml(f"{pathIF}InteractionField_social.xml",weight,actionTimes,inactiveTimes,len(dictionary), fieldGroup)
  update_Agentxml(f"{pathA}Agent_social.xml",or_x,or_y, agentGroup, init_positions)

# S_true=make_trajectory(n,n_agents,mode)