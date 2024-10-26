#!/usr/bin/python3
import os
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import sys
import matplotlib.pyplot as plt


# Plot bar charts for different GPU threads per segment
def plotBarChartThreadsPerSegment(inputFile, saveFig=True, preview=True):
  rawdata = pd.read_csv(inputFile)
  datasets = rawdata.Dataset.unique()
  rawdata.ThreadsPerCluster = rawdata.ThreadsPerCluster.astype(int)
  threads = rawdata.ThreadsPerCluster.unique()
  # generate figures with each dataset
  for dataset in datasets:
    selected = rawdata.loc[rawdata['Dataset'] == dataset]
    # drop the dataset column and redundant thread nums
    selected = selected.drop(columns=['Dataset'])
    selected = selected.drop(selected[selected['ThreadsPerCluster'] == 2].index)
    selected = selected.drop(selected[selected['ThreadsPerCluster'] == 36].index)
    selected = selected.drop(selected[selected['ThreadsPerCluster'] == 40].index)
    selected = selected.drop(selected[selected['ThreadsPerCluster'] == 80].index)
    selected = selected.drop(selected[selected['ThreadsPerCluster'] == 100].index)

    # reorganize the data frame 
    melted_df = pd.melt(selected, id_vars=['ThreadsPerCluster'], var_name='Relation', value_name='Time', 
                        value_vars=['EF', 'ET', 'FT', 'VF', 'VV', 'VE', 'VT'])
    # print(melted_df)

    sns.set_theme(style='whitegrid', font='Arial', font_scale=2)
    fig, sub = plt.subplots(num=dataset)
    sns.barplot(ax=sub, x='Relation', y='Time', hue='ThreadsPerCluster', data=melted_df, palette=sns.color_palette('Set2'))
    sub.set_ylabel("Time (s)")
    plt.legend(title="GPU threads per segment", ncol=2, fancybox=True, loc='upper right')
    sub.set_xlabel("")
    fig.set_figheight(9)
    fig.set_figwidth(16)
    fig.tight_layout(pad=fig_padding)
    fig.set_dpi(150)
    if saveFig:
      prefix = Path(inputFile).stem
      if not os.path.exists("figures/threadsPerSegment"):
        os.makedirs("figures/threadsPerSegment/")
      fig.savefig("figures/threadsPerSegment/" + dataset + ".png")
    if preview:
      plt.show()

# Plot bar charts for different block numbers 
def plotBarChartBlockNum(inputFile, saveFig=True, preview=True):
  rawdata = pd.read_csv(inputFile)
  rawdata['Block num'] = rawdata['Block num'].astype(int)
  rawdata['Block size'] = rawdata['Block size'].astype(int)
  # drop waiting time columns
  rawdata = rawdata[rawdata.columns.drop(list(rawdata.filter(regex='waiting')))]
  blockSizes = rawdata['Block size'].unique()
  # change the unit of memory usage to GB
  rawdata['Memory usage'] = rawdata['Memory usage'] / 1024.0

  # generate figures 
  selected = rawdata.loc[rawdata['Block size'] == 512]
  # print(selected)
  metrics = ['Execution time', 'Memory usage']
  for metric in metrics: 
    # reorganize the data frame 
    # melted_df = pd.melt(selected, id_vars=['ThreadsPerCluster'], var_name='Relation', value_name='Time', 
    #                     value_vars=['EF', 'ET', 'FT', 'VT', 'VV', 'VE', 'VF'])
    # print(melted_df)

    sns.set_theme(style='whitegrid', font='Arial', font_scale=font_zoom)
    fig, sub = plt.subplots()
    sns.barplot(ax=sub, x='Dataset', y=metric, hue='Block num', data=selected, palette=sns.color_palette('Set2'))
    if "time" in metric: 
      sub.set_ylabel(metric + " (s)")
    elif "Memory" in metric: 
      sub.set_ylabel(metric + " (GB)")

    plt.legend(title="GPU block numbers", ncol=2, fancybox=True)
    sub.set_xlabel("")
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)
    fig.tight_layout(pad=fig_padding)
    fig.set_dpi(300)
    if saveFig:
      prefix = Path(inputFile).stem
      if not os.path.exists("figures/gpu-block-num"):
        os.makedirs("figures/gpu-block-num/")
      figure_name = inputFile[:inputFile.find('-')]
      if "time" in metric: 
        figure_name += '-time'
      elif "Memory" in metric: 
        figure_name += '-memory'
      fig.savefig("figures/gpu-block-num/" + figure_name + ".png")
    if preview:
      plt.show()

# Plot bar charts for different block sizes 
def plotBarChartBlockSize(inputFile, saveFig=True, preview=True):
  rawdata = pd.read_csv(inputFile)
  rawdata['Block num'] = rawdata['Block num'].astype(int)
  rawdata['Block size'] = rawdata['Block size'].astype(int)
  # drop waiting time columns
  rawdata = rawdata[rawdata.columns.drop(list(rawdata.filter(regex='waiting')))]
  blockNums = rawdata['Block num'].unique()
  # change the unit of memory usage to GB
  rawdata['Memory usage'] = rawdata['Memory usage'] / 1024.0

  # generate figures 
  selected = rawdata.loc[rawdata['Block num'] == blockNums[0]]
  # print(selected)
  metrics = ['Execution time', 'Memory usage']
  for metric in metrics: 
    sns.set_theme(style='whitegrid', font='Arial', font_scale=font_zoom)
    fig, sub = plt.subplots()
    sns.barplot(ax=sub, x='Dataset', y=metric, hue='Block size', data=selected, palette=sns.color_palette('Set2'))
    if "time" in metric: 
      sub.set_ylabel(metric + " (s)")
    elif "Memory" in metric: 
      sub.set_ylabel(metric + " (GB)")

    plt.legend(title="Thread numbers per GPU block", ncol=3, fancybox=True)
    sub.set_xlabel("")
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)
    fig.tight_layout(pad=fig_padding)
    fig.set_dpi(300)
    if saveFig:
      prefix = Path(inputFile).stem
      if not os.path.exists("figures/gpu-block-size"):
        os.makedirs("figures/gpu-block-size/")
      figure_name = inputFile[:inputFile.find('-')]
      if "time" in metric: 
        figure_name += ('-time-' + str(blockNums[0]) + '-blocks')
      elif "Memory" in metric: 
        figure_name += ('-memory-' + str(blockNums[0]) + '-blocks')
      fig.savefig("figures/gpu-block-size/" + figure_name + ".png")
    if preview:
      plt.show()

# Plot bar charts for different CPU threads
def plotBarChartThreadNum(inputFile, saveFig=True, preview=True):
  rawdata = pd.read_csv(inputFile)
  rawdata['Thread num'] = rawdata['Thread num'].astype(int)
  rawdata['Block size'] = rawdata['Block size'].astype(int)
  # drop waiting time columns
  blockSizes = rawdata['Block size'].unique()
  # change the unit of memory usage to GB
  rawdata['Memory usage'] = rawdata['Memory usage'] / 1024.0
  # drop the single-thread execution time
  rawdata = rawdata.drop(rawdata[rawdata['Thread num'] == 1].index)


  # generate figures 
  for blockSize in blockSizes: 
    selected = rawdata.loc[rawdata['Block size'] == blockSize]
    # print(selected)
    metrics = ['Execution time', 'Memory usage']
    for metric in metrics: 
      sns.set_theme(style='whitegrid', font='Arial', font_scale=font_zoom)
      fig, sub = plt.subplots()
      sns.barplot(ax=sub, x='Dataset', y=metric, hue='Thread num', data=selected, palette=sns.color_palette('Set2'))
      if "time" in metric:
        sub.set_ylabel(metric + " (s)")
      elif "Memory" in metric:
        sub.set_ylabel(metric + " (GB)")

      plt.legend(title="CPU thread numbers", ncol=3, fancybox=True)
      sub.set_xlabel("")
      fig.set_figheight(fig_height)
      fig.set_figwidth(fig_width)
      fig.tight_layout(pad=fig_padding)
      fig.set_dpi(300)
      if saveFig:
        prefix = Path(inputFile).stem
        if not os.path.exists("figures/cpu-thread-num"):
          os.makedirs("figures/cpu-thread-num/")
        figure_name = inputFile[:inputFile.find('-')]
        if "time" in metric: 
          figure_name += ('-time-' + str(blockSize))
        elif "Memory" in metric: 
          figure_name += ('-memory-' + str(blockSize))
        fig.savefig("figures/cpu-thread-num/" + figure_name + ".png")
      if preview:
        plt.show()

# Plot bar charts for comparison 
def plotBarChartComparison(inputFile, saveFig=True, preview=True):
  rawdata = pd.read_csv(inputFile)
  # drop waiting time columns
  rawdata = rawdata[rawdata.columns.drop(list(rawdata.filter(regex='waiting')))]
  # change the unit of memory usage to GB
  rawdata['Memory usage'] = rawdata['Memory usage'] / 1024.0

  # generate figures 
  # metrics = ['Total time', 'Memory usage']
  metrics = ['Execution time']
  for metric in metrics: 
    sns.set_theme(style='whitegrid', font='Arial', font_scale=font_zoom)
    fig, sub = plt.subplots()
    sns.barplot(ax=sub, x='Dataset', y=metric, hue='Data structure', data=rawdata, palette=sns.color_palette('Set2'))
    if "time" in metric: 
      sub.set_ylabel(metric + " (s)")
    elif "Memory" in metric: 
      sub.set_ylabel(metric + " (GB)")

    plt.legend(title="Data structures", ncol=1, fancybox=True)
    sub.set_xlabel("")
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)
    fig.tight_layout(pad=fig_padding)
    fig.set_dpi(300)
    if saveFig:
      prefix = Path(inputFile).stem
      if not os.path.exists("figures/comparison"):
        os.makedirs("figures/comparison/")
      figure_name = inputFile[:inputFile.find('-')]
      # if "time" in metric: 
      #   figure_name += '-time'
      # elif "Memory" in metric: 
      #   figure_name += '-memory'
      figure_name += ('-' + metric.split()[0])
      fig.savefig("figures/comparison/" + figure_name + ".png")
    if preview:
      plt.show()


# Main entry of the program
if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Usage: python3", sys.argv[0], " <type> <input_csv_file>")
    print("Type: ")
    print("1: draw the bar plot of different GPU threads per segment")
    print("2: draw the bar plot of different GPU block numbers")
    print("3: draw the bar plot of different GPU block sizes")
    print("4: draw the bar plot of different CPU thread numbers")
    print("5: draw the comparison bar plot of different data structures")
    sys.exit(-1)

  datasets = ["Engine", "Fish", "Foot", "Asteroid", "Hole", "Stent"]

  # color palettes
  colorScheme = ['#332288', '#117733', '#44AA99', '#88CCEE', '#DDCC77', '#CC6677', '#AA4499', '#882255']

  # parameters for the generated figure
  fig_height = 6
  fig_width = 8
  fig_padding = 0.5
  font_zoom = 1.8

  # generate different figures based on the input
  type = int(sys.argv[1])
  if type == 1:
    plotBarChartThreadsPerSegment(sys.argv[2], True, False)
  elif type == 2:
    plotBarChartBlockNum(sys.argv[2], True, False)
  elif type == 3:
    plotBarChartBlockSize(sys.argv[2], True, False)
  elif type == 4:
    plotBarChartThreadNum(sys.argv[2], True, True)
  elif type == 5:
    plotBarChartComparison(sys.argv[2], True, False)
  else:
    print("Error: wrong drawing type")
    sys.exit(1)

  print("Finished!\n")