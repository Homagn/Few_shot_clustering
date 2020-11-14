import csv
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 18})
import sys
from statistics import mean 
from collections import deque
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser("Few shot clustering for residential occupancy detection")
    # Environment
    parser.add_argument("--file", type=str, help="name of the zone where you want to do occupancy clustering")

    args = parser.parse_args()
    return args.file

def get_values(file_name,pos): #Assuming you want to get some column from this csv file
    val=[]
    with open(file_name, "r") as f:
        for line in f:
            csv_row = line.split() #returns a list ["1","50","60"]
            #print(csv_row)
            #sys.exit(0)
            try:
                l=csv_row[0].split(',')
                #print(l[pos])
                val.append(float(l[pos]))
            except:
                print("Error")
    return val
def plot_column(fname,cn,savename):
    y = get_values(fname,cn)
    print(y)
    x = np.arange(len(y))
    x = x.tolist()
    plt.plot(x,y,linewidth=3)
    plt.xlabel('Number of epochs in multiples of 10')
    plt.ylabel('expected correct hits')
    plt.hold(True)
    #plt.savefig('plot_'+savename+'.png')

def edit(file_name,col_pos,values,new_file_name):#values is the list you want to add, col_pos is the position where you wanna add
    counter=0
    fname = new_file_name
    file1 = open(fname, 'a')
    writer = csv.writer(file1)
    with open(file_name, "r") as f:
        for line in f:
            csv_row = line.split() #returns a list ["1","50","60"]
            #print(csv_row)
            try:
                l=csv_row[0].split(',')
                l.insert(col_pos,values[counter])
                fields1=l
                writer.writerow(fields1)
            except:
                print("Could not write")
            counter+=1
    file1.close()

def episode_accm_perf(cur,tar,dones):
    perf=[]
    p_l=[]
    ep_l=[]
    ep=0
    for i in range(len(dones)):
        #p=float(float(cur[i])/float(tar[i]))
        p=float(cur[i])
        p_l.append(p)
        ep+=1
        if(dones[i]=='TRUE'):
            perf.append(mean(p_l))
            p_l=[]
            ep_l.append(ep)
    return ep_l,perf

def episode_accm_gap_closing(cur,tar,dones,switches):
    perf=[]
    p_l=[]
    ep_l=[]
    ep=0
    for i in range(len(dones)):
        #p=float(float(cur[i])/float(tar[i]))
        p=float(tar[i])
        if(switches[i]=='head'):
            p=float(cur[i])
        if(switches[i]=='hand'):
            p=float(cur[i])
        else:
            p=float(cur[i])
        p_l.append(p)
        ep+=1
        if(dones[i]=='TRUE'):
            #perf.append(float(float(tar[i])-min(p_l))/(float(tar[i])*len(p_l)))
            perf.append(np.mean(p_l)/(float(tar[i])*len(p_l)))
            #perf.append(min(p_l)/(float(tar[i])))
            #perf.append(min(p_l))
            #perf.append(float(float(tar[i])-min(p_l)))
            p_l=[]
            ep_l.append(ep)
    return ep_l,perf


def episode_accm_rewards(rewards,dones):
    episode_count=0
    add=0
    ep_list=[]
    rsum_list=[]
    for i in range (len(rewards)):
        #print(inside)
        add+=float(rewards[i])
        episode_count+=1
        if(dones[i]=='TRUE'):
            ep_list.append(episode_count)
            #rsum_list.append(add)
            rsum_list.append(add/7)
            add=0
    return ep_list,rsum_list

def moving_avg_filter(rewards,win_sz):
    l = deque(maxlen=win_sz)
    l_m=[]
    for i in rewards:
        l.append(i)
        l_m.append(mean(l))
    return l_m

def plot_rewards(fname):
    rewards=get_values(fname,0)
    dones=get_values(fname,1)
    ep,rs=episode_accm_rewards(rewards,dones)
    #print(ep)
    #print(rs)
    '''
    FOR REAL EXPERIMENT (remove Sim while reading csv)
    '''
    '''
    rs=moving_avg_filter(rs,100)
    plt.plot(ep[:-90],rs[:-90],linewidth=3)
    plt.xlabel('Number of epochs')
    plt.ylabel('Averaged total reward per episode')
    plt.show()
    '''
    '''
    FOR SIMULATION
    '''
    
    rs=moving_avg_filter(rs,1000)
    #rs=moving_avg_filter(rs,20)
    for r in range(len(rs)):
        #rs[r]-=0.7
        rs[r]+=0.0
    plt.plot(ep[0:-10],rs[0:-10],linewidth=3)
    plt.xlabel('Number of epochs')
    plt.ylabel('Averaged total reward per episode')
    plt.hold(True)
    
    #edit('test.csv',2,v,'new_csv.csv')
def plot_performance(fname):
    cur_dist=get_values(fname,2)
    tar_dist=get_values(fname,3)
    dones=get_values(fname,1)
    switches=get_values(fname,4)
    #ep,perf=episode_accm_perf(cur_dist,tar_dist,dones)
    ep,perf=episode_accm_gap_closing(cur_dist,tar_dist,dones,switches)
    #print(ep)
    #print(rs)
    '''
    FOR REAL EXPERIMENT
    '''
    '''
    perf=moving_avg_filter(perf,50)
    for p in range(len(perf)):
        perf[p]=(perf[p]-0.8)*3+0.4
    perf=perf[:-90]
    ep=ep[:-90]
    '''
    '''
    FOR SIMULATION
    '''
    
    perf=moving_avg_filter(perf,1000)
    for p in range(len(perf)):
        #perf[p]=(perf[p]-0.95)*5+0.70
        perf[p]=(perf[p])
    
    plt.plot(ep[:-10],perf[:-10],linewidth=3)
    plt.xlabel('Number of epochs')
    plt.ylabel('Averaged tracking error')
    plt.legend()
    plt.hold(True)  # hold is on
    

if __name__ == '__main__':
    filen = parse_args()
    
    fname='Instances/'+filen+'/progressNth.csv'
    plot_column(fname,1,"correct_hits") #provide column number
    #plot_column(fname,5,"expected_occupied_accuracy") #provide column number
    #plot_column(fname,6,"expected_unoccupied_accuracy") #provide column number
    

    '''
    fname='Instances/'+filen+'/loss.csv'
    plot_column(fname,0,"loss") #provide column number
    '''
    plt.show()

