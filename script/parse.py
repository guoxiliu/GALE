#!/usr/bin/python3
import csv
import numpy as np
import os
import pandas as pd
import re
import sys

waitPattern = re.compile(r'missing counts: (\d+), waiting time: (\d+\.\d+);')

def parseCPResults(inputFile, outputFile):
  """
  Parse the results of the Critical Points plugin.
  """
  allData = []
  maxThreadNumber = 0

  # read the file
  with open(inputFile, "r") as fp:
    header = ["Dataset", "Threshold", "Thread num", "Block num", "Block size", "Preprocessing", "Execution time", "Memory usage"]
    info = []
    counts = []
    times = []
    for line in fp:
      if line.startswith("Dataset"):
        dataset = line[line.find(' ')+1 : -1]
        if len(info) > 0:
          if len(counts) > 0:
            info.extend(counts)
          if len(times) > 0:
            info.extend(times)
          allData.append(info)
        info = [dataset]
        counts = []
        times = []
      elif line.startswith("Threshold: "):
        info.append(int(line[line.find(': ')+1 : -1]))
      elif line.startswith("Block num: "):
        info.append(int(line[line.find(': ')+1 : -1]))
      elif line.startswith("Block size: "):
        info.append(int(line[line.find(': ')+1 : -1]))
      elif line.startswith("Thread num: "):
        threadNum = int(line[line.find(': ')+1 : line.find(',')])
        info.append(threadNum)
        maxThreadNumber = max(maxThreadNumber, threadNum)
      elif "Time usage for preconditioning" in line:
        info.append(float(line[line.find(': ')+1 : -3]))
      elif "Processed " in line and "vertices " in line: 
        info.append(float(line[line.rfind('vertices')+8 : -3]))
        # info.append(info[-1] + info[-2])
      elif line.startswith("[ScalarFieldCriticalPoints] Memory usage"):
        info.append(float(line[line.find(':')+1 : -4]))
      elif "missing counts" in line:
        totalCounts = 0
        totalWaitingTime = 0.0
        matches = waitPattern.findall(line)
        for m in matches:
          totalCounts += int(m[0])
          totalWaitingTime += float(m[1])
        counts.append(totalCounts)
        times.append(totalWaitingTime)

    # take care of the last bucket threshold
    if len(info) > 0:
      if len(counts) > 0:
        info.extend(counts)
      if len(times) > 0:
        info.extend(times)
      allData.append(info)

    if 'waiting' in inputFile:
      waitingHeader = ['count ' + str(i) for i in range(maxThreadNumber)]
      header.extend(waitingHeader)
      waitingHeader = ['time ' + str(i) for i in range(maxThreadNumber)]
      header.extend(waitingHeader)

  # write the csv file
  with open(outputFile, 'w', newline='') as fp:
    csvWriter = csv.writer(fp, delimiter=',')
    csvWriter.writerow(header)
    csvWriter.writerows(allData)

def parseDGResults(inputFile, outputFile):
  """
  Parse the results of the Discrete Gradient plugin.
  """
  allData = []

  # read the file
  with open(inputFile, "r") as fp:
    header = ["Dataset", "Threshold", "Thread num", "Block size", "Block num", "Preprocessing", "Execution time", "Total time", "Memory usage"]

    info = []
    counts = []
    times = []
    precondTime = 0
    maxThreadNumber = 0
    for line in fp:
      if line.startswith("Dataset"):
        dataset = line[line.find(' ')+1 : -1]
        if len(info) > 0:
          if len(counts) > 0:
            info.extend(counts)
          if len(times) > 0:
            info.extend(times)
          allData.append(info)
        info = [dataset]
        counts = []
        times = []
        precondTime = 0
      elif line.startswith("Threshold: "):
        info.append(int(line[line.find(': ')+1 : -1]))
      elif line.startswith("Block num: "):
        info.append(int(line[line.find(': ')+1 : -1]))
      elif line.startswith("Block size: "):
        info.append(int(line[line.find(': ')+1 : -1]))
      elif line.startswith("Thread num: "):
        threadNum = int(line[line.find(': ')+1 : line.find(',')])
        info.append(threadNum)
        maxThreadNumber = max(maxThreadNumber, threadNum)
      elif "Edges preconditioned in " in line:
        precondTime += float(line[line.find(' in ')+3 : -3])
      elif "Triangles preconditioned in " in line:
        precondTime += float(line[line.find(' in ')+3 : -3])
      elif "Time usage for preprocessing" in line:
        precondTime = float(line[line.find(':')+1 : -3])
      elif "[DiscreteGradient] Built discrete gradient " in line: 
        info.append(precondTime)
        info.append(float(line[line.rfind('[')+1 : line.find('|')-1]))
        info.append(precondTime + info[-1])
      elif line.startswith("[DiscreteGradient] Memory usage:"):
        info.append(float(line[line.find(':')+1 : -4]))
      elif "missing counts" in line:
        totalCounts = 0
        totalWaitingTime = 0.0
        matches = waitPattern.findall(line)
        for m in matches:
          totalCounts += int(m[0])
          totalWaitingTime += float(m[1])
        counts.append(totalCounts)
        times.append(totalWaitingTime)

    # take care of the last bucket threshold
    if len(info) > 0:
      if len(counts) > 0:
        info.extend(counts)
      if len(times) > 0:
        info.extend(times)
      allData.append(info)

    if 'waiting' in inputFile:
      waitingHeader = ['count ' + str(i) for i in range(maxThreadNumber)]
      header.extend(waitingHeader)
      waitingHeader = ['time ' + str(i) for i in range(maxThreadNumber)]
      header.extend(waitingHeader)

  # write the csv file
  with open(outputFile, 'w', newline='') as fp:
    csvWriter = csv.writer(fp, delimiter=',')
    csvWriter.writerow(header)
    csvWriter.writerows(allData)

def parseFGResults(inputFile, outputFile):
  """
  Parse the results of the Forman Gradient plugin.
  """
  allData = []

  # read the file
  with open(inputFile, "r") as fp:
    if "cache_" in inputFile: 
      header = ["Dataset", "Cache size", "Preprocessing", "Computing", "Execution time", "Total time", "Memory usage"]
    elif "thread_" in inputFile: 
      header = ["Dataset", "Thread num", "Preprocessing", "Computing", "Execution time", "Total time", "Memory usage"]
    elif "mode_" in inputFile:
      header = ["Dataset", "Work mode", "Preprocessing", "Computing", "Execution time", "Total time", "Memory usage"]

    if "producer" in inputFile:
      header.append("Waiting time")
    
    info = []
    for line in fp:
      if line.startswith("Dataset"):
        dataset = line[line.find(' ')+1 : -1]
        if len(info) == len(header):
          allData.append(info)
        info = [dataset]
      elif line.startswith("Cache size"):
        info.append(float(line[line.find(':')+1 : -1]))
      elif line.startswith("Thread num"):
        info.append(int(line[line.find(':')+1 : -1]))
      elif line.startswith("Work mode"):
        info.append(int(line[line.find(':')+1 : -1]))
      elif "Time usage for preconditioning" in line:
        info.append(float(line[line.find(':')+1 : -3]))
      elif "Time usage for computing gradient" in line: 
        info.append(float(line[line.find(':')+1 : -3]))
        info.append(info[-1] + info[-2])
      elif line.startswith("[FormanGradient] Memory usage"):
        info.append(float(line[line.find(':')+1 : -4]))
      elif line.startswith("waiting time: "):
        info.append(float(line[line.find(':')+1:]))

    # take care of the last bucket threshold
    if len(info) == len(header):
      allData.append(info)

  # write the csv file
  with open(outputFile, 'w', newline='') as fp:
    csvWriter = csv.writer(fp, delimiter=',')
    csvWriter.writerow(header)
    csvWriter.writerows(allData)

def parseFTMResults(inputFile, outputFile):
  """
  Parse the results of the FTM tree plugin.
  """
  allData = []
  maxThreadNumber = 0

  # read the file
  with open(inputFile, "r") as fp:
    header = ["Dataset", "Block num", "Block size", "Thread num", "Execution time", "Memory usage"]

    info = []
    for line in fp:
      if line.startswith("Dataset"):
        dataset = line[line.find(' ')+1 : -1]
        if len(info) > 0:
          allData.append(info)
        info = [dataset]
      elif line.startswith("Block num: "):
        info.append(int(line[line.find(': ')+1 : -1]))
      elif line.startswith("Block size: "):
        info.append(int(line[line.find(': ')+1 : -1]))
      elif line.startswith("Thread num: "):
        threadNum = int(line[line.find(': ')+1 : line.find(',')])
        info.append(threadNum)
        maxThreadNumber = max(maxThreadNumber, threadNum)
      elif "Time usage for preconditioning" in line:
        info.append(float(line[line.find(': ')+1 : -3]))
      elif line.startswith("[FTMTree_MT] Total"):
        info.append(float(line[line.find(': ')+1 : line.find(',')]))
        # info.append(info[-1] + info[-2])
      elif line.startswith("[FTMTree] Memory usage"):
        info.append(float(line[line.find(':')+1 : -4]))
      # elif line.startswith("waiting count"):
      #   info.append(float(line[line.rfind(': ')+1 : -1]))

    # take care of the last bucket threshold
    if len(info) > 0:
      allData.append(info)

    # waitingHeader = ['waiting ' + str(i) for i in range(maxThreadNumber)]
    # header.extend(waitingHeader)

  # write the csv file
  with open(outputFile, 'w', newline='') as fp:
    csvWriter = csv.writer(fp, delimiter=',')
    csvWriter.writerow(header)
    csvWriter.writerows(allData)

def parseTRResults(inputFile, outputFile):
  """
  Parse the results of the Forman Gradient plugin.
  """
  
  allData = []
  header = ["Dataset", "Block num", "Block size", "Edge precondition", "Triangle precondition", "EF", "ET", "FT", "VV", "VE", "VF", "VT"]
  # header = ["Dataset", "Block num", "Block size", "Thread num", "Edge precondition", "Triangle precondition", "EF", "ET", "FT", "VV", "VE", "VF", "VT"]
  if 'actopo' in inputFile:
    header = ["Dataset", "Thread num", "Edge precondition", "Triangle precondition", "EF", "ET", "FT", "VV", "VE", "VF", "VT"]
#   header = ["Dataset", "Preprocessing", "VT", "EV", "ET", "FV", "FT", "TV", "VE", "VF", "EF", "FE", "TE", "TF", "Execution time", "Total time", "Memory usage"]

  # read the file
  with open(inputFile, "r") as fp:
    info = []
    for line in fp:
      if line.startswith("Dataset"):
        dataset = line[line.find(' ')+1 : -1]
        if len(info) > 0:
          # info[-2] += info[2]
          allData.append(info)
        info = [dataset]
      # elif line.startswith('Cluster number:'):
      #   clusters = int(line[line.find(':')+1 : -1])
      #   info.append(clusters)
      # elif line.startswith("Parsing vtu file"):
      #   dataset = line[line.rfind('/')+1 : line.rfind('_')]
      #   if len(info) > 0:
      #     allData.append(info)
      #   info = [dataset]
      elif line.startswith("Block num: "):
        info.append(int(line[line.find(': ')+1 : -1]))
      elif line.startswith("Block size: "):
        info.append(int(line[line.find(': ')+1 : -1]))
      elif line.startswith("Thread num: "):
        threadNum = int(line[line.find(': ')+1 : line.find(',')])
        info.append(threadNum)
      elif "Edges preconditioned in " in line:
        info.append(float(line[line.find(' in ')+3 : -3]))
      elif "Triangles preconditioned in " in line:
        info.append(float(line[line.find(' in ')+3 : -3]))
      elif "Time usage for " in line: 
        if "Time usage for preconditioning:" in line:
          continue
        info.append(float(line[line.find(':')+1 : -3]))
      # elif "[TestTopoCluster] Data-set" and " processed in " in line: 
      #   info.append(float(line[line.find(' in ')+4 : line.find('s.')]))
      # elif line.startswith("[TestTopoCluster] Memory usage"):
      #   info.append(float(line[line.find(':')+1 : -4]))
      # elif line.startswith("[ttkTestTopoCluster] Memory usage"):
      #   info.append(float(line[line.find(':')+1 : -4]))
      # elif line.startswith("waiting time: "):
      #   info.append(float(line[line.find(':')+1:]))
      # elif line.startswith("buffer misses: "):
      #   info.append(float(line[line.find(':')+1:]))
      # elif line.startswith("waiting count"):
      #   info.append(int(line[line.find(': ')+1 : -1]))

    # take care of the last entry
    if len(info) > 0:
      # info[-2] += info[2]
      allData.append(info)

  # write the csv file
  with open(outputFile, 'w', newline='') as fp:
    csvWriter = csv.writer(fp, delimiter=',')
    csvWriter.writerow(header)
    csvWriter.writerows(allData)

def parseMSResults(inputFile, outputFile):
  """
  Parse the results of the Morse-Smale complex plugin.
  """
  header = ["Dataset", "Block num", "Block size", "Thread num", "Preprocessing", "Discrete gradient", "Descending 1-separatrices", "Ascending 1-separatrices", "Saddle connectors", "1-separatrices set", "Execution time", "Valid time", "Total time", "Memory usage"]

  if "waiting" in inputFile:
    header.append("Waiting time")
    header.append("Buffer misses")
  
  allData = []

  # read the file
  with open(inputFile, "r") as fp:
    info = []
    for line in fp:
      if line.startswith("Dataset"):
        dataset = line[line.find(' ')+1 : -1]
        # if len(info) >= len(header):
        if len(info) > 0:
          allData.append(info)
        info = [dataset]
        precondTime = 0
      elif line.startswith("Block num: "):
        info.append(int(line[line.find(': ')+1 : -1]))
      elif line.startswith("Block size: "):
        info.append(int(line[line.find(': ')+1 : -1]))
      elif line.startswith("Thread num"):
        info.append(int(line[line.find(':')+1 : -1]))
      elif "Edges preconditioned in " in line:
        precondTime += float(line[line.find(' in ')+3 : -3])
      elif "Triangles preconditioned in " in line:
        precondTime += float(line[line.find(' in ')+3 : -3])
      elif "Time usage for preconditioning" in line:
        precondTime = float(line[line.find(':')+1 : -3])
      elif "Discrete gradient computed" in line:
        info.append(precondTime)
        info.append(float(line[line.rfind('[')+1 : line.find('|')-1]))
      elif "Descending 1-separatrices computed" in line: 
        info.append(float(line[line.rfind('[')+1 : line.find('|')-1]))
      elif "Ascending 1-separatrices computed" in line: 
        info.append(float(line[line.rfind('[')+1 : line.find('|')-1]))
      elif "Saddle connectors computed" in line: 
        info.append(float(line[line.rfind('[')+1 : line.find('|')-1]))
      elif "1-separatrices set" in line: 
        info.append(float(line[line.rfind('[')+1 : line.find('|')-1]))
      elif "Segmentation computed" in line: 
        info.append(float(line[line.rfind('[')+1 : line.find('|')-1]))
      elif "[MorseSmaleComplex3D] Data-set " and " processed ." in line: 
        info.append(float(line[line.rfind('[')+1 : line.find('|')-1]))
        info.append(info[-1] - info[5])   # valid time (minus discrete gradient)
        info.append(info[-1] + info[4])   # valid time + preprocessing time
        # info.append(float(line[line.rfind('[')+1 : line.find('|')-1]) - dgTime)
      elif "MorseSmaleComplex] Memory usage" in line:
        info.append(float(line[line.find(':')+1 : -4]))
      elif line.startswith("waiting time: "):
        info.append(float(line[line.find(':')+1:]))
      elif line.startswith("buffer misses: "):
        info.append(float(line[line.find(':')+1:]))

    # take care of the last bucket threshold
    # if len(info) >= len(header):
    if len(info) > 0:
      allData.append(info)

  # write the csv file 
  with open(outputFile, 'w', newline='') as fp:
    csvWriter = csv.writer(fp, delimiter=',')
    csvWriter.writerow(header)
    csvWriter.writerows(allData)


def parseQueueStat(inputFile, outputFile, datafile):
  """
  Parse the queue statistics.
  """
  allData = []

  # read the file
  with open(inputFile, "r") as fp:
    # header = ["Dataset", "Threshold", "Thread num", "Block num", "Execution time", "Consumer time", "Queue size", "Queue time", "Preparation time", "Kernel time", "Return time", "Total consumer time", "Total queue time", "Total preparation time", "Total kernel time", "Total return time"]
    header = ["Dataset", "Threshold", "Thread num", "Block num", "Execution time", "Memory usage", "Queue size", "Queue time", "Preparation time", "Kernel time", "Post-kernel time", "Total queue time", "Total preparation time", "Total kernel time", "Total post-kernel time"]
    threadNum = 1
    blockNum = 4
    info = []
    lengths = []
    times = []
    kernelTimes = []
    returnTimes = []
    datamap = {}
    for line in fp:
      if line.startswith("Dataset"):
        dataset = line[line.find(' ')+1 : -1]
        if len(info) > 0:
          if len(lengths) > 0:
            # print("length of the queue statistics:", len(lengths))
            prepareTimes = [x-y-z for x,y,z in zip(times, kernelTimes, returnTimes)]
            info.append(sum(lengths) / len(lengths))
            info.append(sum(times) / len(times))
            info.append(sum(prepareTimes) / len(prepareTimes))
            info.append(sum(kernelTimes) / len(kernelTimes))
            info.append(sum(returnTimes) / len(returnTimes))
            info.append(sum(times))
            info.append(sum(prepareTimes))
            info.append(sum(kernelTimes))
            info.append(sum(returnTimes))
            datamap['lengths-'+str(threadNum)+'-'+str(blockNum)] = lengths
            datamap['times-'+str(threadNum)+'-'+str(blockNum)] = times
            datamap['kernelTimes-'+str(threadNum)+'-'+str(blockNum)] = kernelTimes
            datamap['returnTimes-'+str(threadNum)+'-'+str(blockNum)] = returnTimes
          if len(consumerTimes) > 0:
            datamap['consumerTimes-'+str(threadNum)+'-'+str(blockNum)] = consumerTimes
            # np.savez('cp-queue-' + str(threadNum) + '.npz', array1=lengths, array2=times)
          allData.append(info)
        info = [dataset]
        lengths = []
        times = []
        consumerTimes = []
        kernelTimes = []
        returnTimes = []
      elif line.startswith("Threshold: "):
        info.append(int(line[line.find(': ')+1 : -1]))
      elif line.startswith("Block num: "):
        blockNum = int(line[line.find(': ')+1 : -1])
        info.append(blockNum)
      elif line.startswith("Thread num: "):
        threadNum = int(line[line.find(': ')+1 : line.find(',')])
        info.append(threadNum)
      elif line.startswith('consumer'):
        consumerTimes.append(float(line[line.find(':')+1 : -1]))
      elif line.startswith("queue length"):
        lengths.append(float(line[line.find(':')+1 : line.find(',')]))
        if 'total time:' in line: 
          times.append(float(line[line.find('total time:')+11 : line.find(', kernel time')]))
        else:
          times.append(float(line[line.find('time:')+5 : -1]))
        if 'kernel time:' in line: 
          kernelTimes.append(float(line[line.find('kernel time:')+12 : line.find(', return time')]))
        if 'return time:' in line: 
          returnTimes.append(float(line[line.find('return time:')+12 : -1]))
      # elif "Processed " in line and "vertices " in line: 
      #   info.append(float(line[line.rfind('vertices')+8 : -3]))
      elif "Time usage for computation" in line: 
        info.append(float(line[line.find(': ')+1 : -3]))
      elif line.startswith("[ScalarFieldCriticalPoints] Memory usage"):
        info.append(float(line[line.find(':')+1 : -4]))
      elif "[DiscreteGradient] Built discrete gradient " in line: 
        info.append(float(line[line.rfind('[')+1 : line.find('|')-1]))

    # take care of the last bucket threshold
    if len(info) > 0:
      if len(lengths) > 0:
        # print("length of the queue statistics:", len(lengths))
        prepareTimes = [x-y-z for x,y,z in zip(times, kernelTimes, returnTimes)]
        info.append(sum(lengths) / len(lengths))
        info.append(sum(times) / len(times))
        info.append(sum(prepareTimes) / len(prepareTimes))
        info.append(sum(kernelTimes) / len(kernelTimes))
        info.append(sum(returnTimes) / len(returnTimes))
        info.append(sum(times))
        info.append(sum(prepareTimes))
        info.append(sum(kernelTimes))
        info.append(sum(returnTimes))
        datamap['lengths-'+str(threadNum)+'-'+str(blockNum)] = lengths
        datamap['times-'+str(threadNum)+'-'+str(blockNum)] = times
        datamap['kernelTimes-'+str(threadNum)+'-'+str(blockNum)] = kernelTimes
        datamap['returnTimes-'+str(threadNum)+'-'+str(blockNum)] = returnTimes
      if len(consumerTimes) > 0:
        datamap['consumerTimes-'+str(threadNum)+'-'+str(blockNum)] = consumerTimes
      allData.append(info)

  # write the csv file
  with open(outputFile, 'w', newline='') as fp:
    csvWriter = csv.writer(fp, delimiter=',')
    csvWriter.writerow(header)
    csvWriter.writerows(allData)
    print(datamap.keys())
    np.savez(datafile, **datamap)


def getAverageValues(filename):
  rawdata = pd.read_csv(filename)
  datasets = rawdata.Dataset.unique()
  ret_avg = pd.DataFrame()
  ret_std = pd.DataFrame()
  if 'simplex' in filename or 'gale' in filename: 
    for dataset in datasets:
      threads = None
      blockNum = None
      blockSize = None
      if 'Thread num' in rawdata:
        threads = rawdata['Thread num'].unique()
      if 'Block num' in rawdata:
        blockNum = rawdata['Block num'].unique()
      if 'Block size' in rawdata:
        blockSize = rawdata['Block size'].unique()
      for bn in blockNum: 
        cur_avg = pd.DataFrame()
        cur_std = pd.DataFrame()

        if blockSize is None and threads is None: 
          selected = rawdata.loc[(rawdata['Dataset'] == dataset) & (rawdata['Block num'] == bn)]
          # print(selected.mean().to_frame().T)
          if not selected.empty:
            average = selected.mean(numeric_only=True).to_frame().T
            deviation = selected.std(numeric_only=True).to_frame().T
            cur_avg = pd.concat([cur_avg, average])
            cur_std = pd.concat([cur_std, deviation])
        
        elif threads is None: 
          for bs in blockSize:
            selected = rawdata.loc[(rawdata['Dataset'] == dataset) & (rawdata['Block num'] == bn) & (rawdata['Block size'] == bs)]
            # print(selected.mean().to_frame().T)
            if not selected.empty:
              average = selected.mean(numeric_only=True).to_frame().T
              deviation = selected.std(numeric_only=True).to_frame().T
              cur_avg = pd.concat([cur_avg, average])
              cur_std = pd.concat([cur_std, deviation])

        else:
          for bs in blockSize:
            for t in threads: 
              selected = rawdata.loc[(rawdata['Dataset'] == dataset) & (rawdata['Thread num'] == t) & (rawdata['Block num'] == bn) & (rawdata['Block size'] == bs)]
              # print(selected.mean().to_frame().T)
              if not selected.empty:
                average = selected.mean(numeric_only=True).to_frame().T
                deviation = selected.std(numeric_only=True).to_frame().T
                cur_avg = pd.concat([cur_avg, average])
                cur_std = pd.concat([cur_std, deviation])
        
        if not cur_avg.empty:
          cur_avg.insert(0, 'Dataset', dataset)
        if not cur_std.empty:
          cur_std.insert(0, 'Dataset', dataset)
        ret_avg = pd.concat([ret_avg, cur_avg])
        ret_std = pd.concat([ret_std, cur_std])

  elif 'block' in filename:
    for dataset in datasets:
      threads = rawdata['Thread num'].unique()
      blocks = rawdata['Block num'].unique()
      for b in blocks: 
        cur_avg = pd.DataFrame()
        cur_std = pd.DataFrame()
        for t in threads: 
          selected = rawdata.loc[(rawdata['Dataset'] == dataset) & (rawdata['Thread num'] == t) & (rawdata['Block num'] == b)]
          # print(selected.mean().to_frame().T)
          if not selected.empty:
            average = selected.mean(numeric_only=True).to_frame().T
            deviation = selected.std(numeric_only=True).to_frame().T
            cur_avg = pd.concat([cur_avg, average])
            cur_std = pd.concat([cur_std, deviation])
        if not cur_avg.empty:
          cur_avg.insert(0, 'Dataset', dataset)
        if not cur_std.empty:
          cur_std.insert(0, 'Dataset', dataset)
        ret_avg = pd.concat([ret_avg, cur_avg])
        ret_std = pd.concat([ret_std, cur_std])
  elif 'waiting' in filename: 
    for dataset in datasets:
      thresholds = rawdata['Threshold'].unique()
      threads = rawdata['Thread num'].unique()
      blockNum = rawdata['Block num'].unique()
      blockSize = rawdata['Block size'].unique()
      for thres in thresholds:
        for bn in blockNum: 
          for bs in blockSize:
            cur_avg = pd.DataFrame()
            cur_std = pd.DataFrame()
            for t in threads: 
              selected = rawdata.loc[(rawdata['Dataset'] == dataset) & (rawdata['Threshold'] == thres) & (rawdata['Thread num'] == t) & (rawdata['Block num'] == bn) & (rawdata['Block size'] == bs)]
              # print(selected.mean().to_frame().T)
              if not selected.empty:
                average = selected.mean(numeric_only=True).to_frame().T
                deviation = selected.std(numeric_only=True).to_frame().T
                cur_avg = pd.concat([cur_avg, average])
                cur_std = pd.concat([cur_std, deviation])
            if not cur_avg.empty:
              cur_avg.insert(0, 'Dataset', dataset)
            if not cur_std.empty:
              cur_std.insert(0, 'Dataset', dataset)
            ret_avg = pd.concat([ret_avg, cur_avg])
            ret_std = pd.concat([ret_std, cur_std])
  else: 
    for dataset in datasets:
      cur_avg = pd.DataFrame()
      cur_std = pd.DataFrame()
      threads = rawdata['Thread num'].unique()
      for t in threads:
        selected = rawdata.loc[(rawdata['Dataset'] == dataset) & (rawdata['Thread num'] == t)]
        # print(selected.mean().to_frame().T)
        if not selected.empty:
          average = selected.mean(numeric_only=True).to_frame().T
          deviation = selected.std(numeric_only=True).to_frame().T
          cur_avg = pd.concat([cur_avg, average])
          cur_std = pd.concat([cur_std, deviation])
      if not cur_avg.empty:
        cur_avg.insert(0, 'Dataset', dataset)
      if not cur_std.empty:
        cur_std.insert(0, 'Dataset', dataset)
      ret_avg = pd.concat([ret_avg, cur_avg])
      ret_std = pd.concat([ret_std, cur_std])
      # ret_avg = ret_avg.append(cur_avg)
  
  # save output to the csv file
  ret = pd.concat([ret_avg])
  # ret = pd.concat([ret_avg, ret_std])
  outFile = os.path.splitext(filename)[0] + '-avg.csv'
  ret.to_csv(outFile, index=False)


def parseGPUProfile(filename):
  pattern = re.compile(r'^\s*([A-Za-z\s]*:)*\s*(\d+\.\d+%)\s+([\d\.]+[a-z]*)\s+([\d,]+)\s+([\d\.]+[a-z]*)\s+([\d\.]+[a-z]*)\s+([\d\.]+[a-z]*)\s+(.*)$')
  data = []
  header = ['Dataset', 'Threshold', 'Block num', 'Block size', 'Thread num', 'Type', 'Time(%)', 'Time', 'Calls', 'Avg', 'Min', 'Max', 'Name']

  # Parse text
  with open(filename, "r") as fp:
    current_type = None
    info = []
    for line in fp:
      typeline = False
      if line.startswith("Dataset"):
        dataset = line[line.find(' ')+1 : -1]
        if len(info) >= len(header):
          data.append(info)
        info = [dataset]
      elif line.startswith("Threshold: "):
        info.append(int(line[line.find(': ')+1 : -1]))
      elif line.startswith("Block num: "):
        info.append(int(line[line.find(': ')+1 : -1]))
      elif line.startswith("Block size: "):
        info.append(int(line[line.find(': ')+1 : -1]))
      elif line.startswith("Thread num"):
        info.append(int(line[line.find(':')+1 : -1]))
      elif line.strip().startswith('GPU activities:'):
        current_type = 'GPU activities'
        typeline = True
        # continue
      elif line.strip().startswith('API calls:'):
        current_type = 'API calls'
        # continue
      match = pattern.match(line)
      if match:
        # if typeline: 
          data.append([*info, current_type, *match.groups()[1:]])
        # else: 
          # data.append([*info, current_type, *match.groups()])

  # Write to CSV file
  outFile = os.path.splitext(filename)[0] + '-GPU.csv'
  with open(outFile, mode='w') as file:
      writer = csv.writer(file, delimiter=',')
      writer.writerow(header)
      writer.writerows(data)

  print(f'Data successfully written to {outFile}')

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Usage: python3", sys.argv[0], "  <type>  <input file>  [output file]  [data file]")
    print("Type: ")
    print("1: parse the Topological Relation results")
    print("2: parse the Critical Points results")
    print("3: parse the FTM Tree results")
    print("4: parse the Discrete Gradient results")
    print("5: parse the Morse-Smale Complex results")
    print("6: get the average values for the csv file")
    print("7: parse the GPU profiling results")
    print("8: parse the queue statistics")
    sys.exit()
  type = int(sys.argv[1])
  inFile = sys.argv[2]

  # write the array as a csv file
  if len(sys.argv) > 3:
    outFile = sys.argv[3]
  else:
    outFile = inFile.rsplit('.', 1)[0] + '.csv'

  if type == 1:
    parseTRResults(inFile, outFile)
  elif type == 2:
    parseCPResults(inFile, outFile)
  elif type == 3:
    parseFTMResults(inFile, outFile)
  elif type == 4:
    parseDGResults(inFile, outFile)
  elif type == 5:
    parseMSResults(inFile, outFile)
  elif type == 6:
    getAverageValues(inFile)
  elif type == 7:
    parseGPUProfile(inFile)
  elif type == 8:
    if len(sys.argv) > 4:
      datafile = sys.argv[4]
    else: 
      datafile = inFile.rsplit('.', 1)[0]
    parseQueueStat(inFile, outFile, datafile)
  else:
    print("Invalid parameter!")
    exit(1)

  print("Finished!\n")

