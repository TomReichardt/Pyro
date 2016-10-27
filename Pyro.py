import matplotlib.cm as cm
import matplotlib.transforms as transforms
import matplotlib.pyplot as P
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline
from scipy.signal import savgol_filter
import numpy as np
import sys, os
from matplotlib.widgets import Slider
from math import log10, floor, ceil
import re
import curses
from ast import literal_eval
import seaborn.apionly as sns
from sklearn.neighbors import KernelDensity
from itertools import izip

#To do list
# - Tunnels (tunnels), wherein data can be input and have specific operations put on them
#       like the timescale of inspiral
# - Add ability to read other types of file (csv etc.)
# - Fix issue to do with changing the colour of a line to a number beyond how many lines there are. Manually make the palettes with 10 colours...
# - Make PEP8...

#P.ion()

def onPress(event):
    global isPicked
    fig = P.gcf()
    ax = P.gca()
    ticker = mpl.ticker.ScalarFormatter(useMathText=True, useOffset=False)
    if isPicked == False:
        if event.key == 'l':
            if event.y < fig.get_size_inches()[1]*fig.dpi * np.array(ax.get_position())[0][1]:
                if ax.get_xscale() == 'linear':
                    ax.set_xscale('log')
                else:
                    ax.set_xscale('linear')
                    ax.xaxis.set_major_formatter(ticker)
                    ax.xaxis.get_major_formatter().set_powerlimits((0,1))

            elif event.x < fig.get_size_inches()[0]*fig.dpi * np.array(ax.get_position())[0][0]:
                if ax.get_yscale() == 'linear':
                    ax.set_yscale('log')
                else:
                    ax.set_yscale('linear')
                    ax.yaxis.set_major_formatter(ticker)
                    ax.yaxis.get_major_formatter().set_powerlimits((0,1))

            ax.set_xlim(min(ax.get_xlim()),max(ax.get_xlim()))
            ax.set_ylim(min(ax.get_ylim()),max(ax.get_ylim()))

            fig.canvas.draw()
        elif event.key == 'a':
            ax.autoscale(tight = 'True')
            fig.canvas.draw()
        elif event.key == 'q':
            P.close()
        elif event.key == 'm':
            global globalPageSetup
            indices = [sns.color_palette(n_colors=10).index(globalLines[i].get_color()) for i, ik in enumerate(globalLines)]

            if globalPageSetup['Palette'] < (len(palettes) - 1): 
                globalPageSetup['Palette'] += 1 
            else:
                globalPageSetup['Palette'] = 0

            print palettes[globalPageSetup['Palette']].keys()[0]

            sns.set_palette(palettes[globalPageSetup['Palette']].values()[0])
            for i,ik in enumerate(globalLines):
                c = sns.color_palette(n_colors=10)[indices[i]]
                globalLines[i].set_color(c)

            if globalLegendSetting == 2:
                ax.legend([ik.get_label() for i, ik in enumerate(globalLines)])

            fig.canvas.draw()
        else:
            print 'Nothing assigned to key \"%s\"!' % event.key
    elif isPicked == True:
        isPicked = False

def onPick(event):
    global isPicked
    fig = P.gcf()
    thisLine = event.artist
    thisLine.set_linewidth(thisLine.get_linewidth() * 2.)
    fig.canvas.draw()
    isPicked = True  
    cid = fig.canvas.mpl_connect('key_press_event', lambda event: onPickPress(event, thisLine, cid))

def onPickPress(event, thisLine, cid):
    fig = P.gcf()
    ax = P.gca()
    key = event.key
    if key in ['1','2','3','4','5','6','7','8','9','0']:
        thisLine.set_color(sns.color_palette(n_colors=10)[int(key)])
    #elif key == 's':
    #    print 'Style can be in (- -- -. :)'
    #    style = defaultInput('Please choose the linestyle desired for this line', '-')
    #    thisLine.set_linestyle(style)
    thisLine.set_linewidth(thisLine.get_linewidth() / 2.)
    if globalLegendSetting == 2:
        ax.legend([ik.get_label() for i, ik in enumerate(globalLines)])
    fig.canvas.draw()
    fig.canvas.mpl_disconnect(cid)

def containsAny(seq, aset):
    for c in seq:
        if c in aset:
            return True
    return False

def defaultInput(message, default, outputType = 'string'):
    var = raw_input(str(message) + ' (default = %s) : ' % str(default))
    if var == '':
        return default
    else:
        if outputType == 'string':
            return str(var)
        elif outputType == 'int':
            return int(var)
        elif outputType == 'float':
            return float(var)

def yesNoSelect(question, default = 'yes'):
    var = raw_input(str(question) + ' (default = %s)? ' % str(default))
    if var == '':
        var = default
    if var in ['y', 'Y', 'yes', 'Yes', 'YES']:
        return True
    elif var in ['n', 'N', 'no', 'No', 'NO']:
        return False

def findNearest(array,value):
    idx = (np.abs(np.array(array)-value)).argmin()
    return idx

def getWinWidth():
    screen = curses.initscr()
    height, width = screen.getmaxyx()
    curses.endwin()
    return width

def readFile():
    filenames = sys.argv[1:]
    openFiles, vals, titles = [], [], []
    for i, ik in enumerate(filenames):
        openFiles.append(open(ik, 'r'))
        vals.append([])
        foundTitles, titleLine = None, None
        for line in openFiles[i].readlines():
            lineList = line.split()
            if (lineList[0][0] == '*')or(lineList[0][0] == '#')or(len(lineList) < 2):
                titleLine = line
                titles.append([[]])
            else:
                for j, jk in enumerate(lineList):
                    try:
                        vals[i][j].append(float(jk))
                    except IndexError:
                        vals[i].append([float(jk)])

        #if len(re.findall(r'\[[\s0-9]*([\w\s]+)\]', titleLine, re.I)) == len(vals[i]):
        #    titles[i][0] = re.findall(r'\[[\s0-9]*([\w\s]+)\]', titleLine, re.I)

        if len(re.findall(r'([a-z./-]{2,}(?:\s?[\w./-]+)*)', titleLine, re.I)) == len(vals[i]):
            titles[i][0] = re.findall(r'([a-z./-]{2,}(?:\s?[\w./-]+)*)', titleLine, re.I)

        if len(titles[i][0]) != len(vals[i]):
            titles[i][0] = ['Column %i' %(j+1) for j, jk in enumerate(vals[i])]

    return vals,titles

def readStyle():
    styleNum = 0
    styleFile = open(os.path.join(os.path.dirname(__file__), "Styles.txt"))
    styles = []
    for line in styleFile.readlines():
        lineList = [x.strip() for x in line.split(':')] 
        if ((len(lineList) == 1 and lineList[0] != '') and lineList[0][0] == '#'):
            styleNum += 1
            styles.append({})
        elif len(lineList) > 1:
            styles[styleNum-1][lineList[0]] = literal_eval(lineList[1])
        else:
            pass
    return styles

def readPalette():
    paletteNum = 0
    paletteNumType = 0
    palettes = []
    try:
        paletteFile = open(os.path.join(os.path.dirname(__file__), "Palettes.txt"))
        for line in paletteFile.readlines():
            if re.match('##', line, re.I) is not None:
                paletteNumType +=1
            if (paletteNumType == 1 and re.match('##', line, re.I) is None and len(line.strip()) != 0):
                palettes.append({line.split(':')[0].strip() : sns.color_palette(re.findall(r'"(#\w+)"', line, re.I))})
                paletteNum += 1         
            elif (paletteNumType == 2 and re.match('##', line, re.I) is None and len(line.strip()) != 0):
                palettes.append({line.split(':')[0].strip() : sns.blend_palette(re.findall(r'"(#\w+)"', line, re.I), 10)})
                paletteNum += 1
            elif (paletteNumType == 3 and re.match('##', line, re.I) is None and len(line.strip()) != 0):
                palettes.append({line.split(':')[0].strip() : sns.light_palette(re.findall(r'"(#\w+)"', line, re.I)[0], 10, reverse=True)})
                paletteNum += 1
    except IOError:
        pass
    return palettes

def readUnits(filePath):
    if filePath == None:
        return
    unitsFile = open(filePath)
    units = {}
    for line in unitsFile.readlines():
        lineList = [x.strip() for x in line.split(':')] 
        if len(lineList) > 1:
            units[str(lineList[0])] = float(lineList[1])
        else:
            pass
    return units

#==============================================================================
# Plotting functions
#==============================================================================

def makePlot(command, currentData, plotType, xVal, yVals=None):

    global globalLims, globalLines

    ax, fig = page()

    maxLims = returnMaxLims(xVal, yVals)

    for i, ik in enumerate(globalLims):
        if globalLims[i] == None:
            globalLims[i] = maxLims[i]
        else:
            pass

    globalLines = [None] * len(yVals)
    for i, ik in enumerate(yVals if len(yVals) > 0 else [None]):
        buildPlot(i, xVal[i], ik, globalLims, plotType)
    inputLegend, inputYVals, inputXVal = list(globalLegend), list(yVals), list(xVal)
    if plotType == 1:
        legend(inputLegend, command, inputYVals, inputXVal, currentData)

    labels(command, currentData, globalLabs)

    isPicked = False
    fig.canvas.mpl_connect('key_press_event', onPress)
    fig.canvas.mpl_connect('pick_event', onPick)
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

    if globalInsetToggle == True:
        global globalInsetLims
        inset(ax, globalInsetLims[4], globalInsetLims[:4], xVal, yVals)

    if globalLegendSetting == 3:
        P.gca().callbacks.connect('xlim_changed', lineLabelShift)
        P.gca().callbacks.connect('ylim_changed', lineLabelShift)


    P.show()


    globalLims = [None if globalLims[i] == maxLims[i] else globalLims[i] for i, ik in enumerate(maxLims)]


def buildPlot(i, plotX, plotY = None, lims = None, plotType = [1,1]):
    global globalLines
    if plotType[0] == 1:
        if smoothed:
            smooth = InterpolatedUnivariateSpline(plotX, plotY, k = 3)
            plotX = np.linspace(min(plotX),max(plotX),num = 10*len(plotX))
            plotY = smooth(plotX)

        lineStyles = ['-', '--', '-.', ':']

        globalLines[i], = P.plot(plotX, plotY, linestyle=lineStyles[int(floor(float((i+1))/len(sns.color_palette())))], picker = 5)

        if not lims:
            lims = [None, None, None, None]
            lims[0] = float(min(plotX))
            lims[1] = float(max(plotX))
            lims[2] = float(min(plotY))
            lims[3] = float(max(plotY))
        limits(lims)

    elif plotType[0] == 2:
        if plotType[1] == 1:
            hist, bins = np.histogram(plotX, bins = 50, range = [lims[0], lims[1]])
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            P.bar(center, hist, align='center', width=width)
        elif plotType[1] != 1:
            if plotType[1] == 2:
                kde = KernelDensity(kernel='gaussian', bandwidth=abs(lims[0] - lims[1]) / 50.).fit(plotX[:,np.newaxis])
            elif plotType[1] == 3:
                kde = KernelDensity(kernel='exponential', bandwidth=abs(lims[0] - lims[1]) / 50.).fit(plotX[:,np.newaxis])
            elif plotType[1] == 4:
                kde = KernelDensity(kernel='tophat', bandwidth=abs(lims[0] - lims[1]) / 50.).fit(plotX[:,np.newaxis])
            xSpace = np.linspace(lims[0]-abs(lims[0]-lims[1])/5., lims[1]+abs(lims[0]-lims[1])/5., 10000)
            logDensity = kde.score_samples(xSpace[:, np.newaxis])
            P.fill(np.concatenate([[min(xSpace)],xSpace,[max(xSpace)]]), np.concatenate([[min(np.exp(logDensity))],np.exp(logDensity),[min(np.exp(logDensity))]]), fc='#AAAAFF')

    elif plotType[0] == 3:
        P.scatter(plotX, plotY)
        limits(lims)


def limits(lims):
    P.xlim(lims[0], lims[1])
    P.ylim(lims[2], lims[3])

def returnMaxLims(xVal, yVals=None):
    if globalPlotType[0] in [1,3]:
        xMin, xMax, yMin, yMax = np.amin(np.hstack(xVal)), np.amax(np.hstack(xVal)), np.amin(np.hstack(yVals)), np.amax(np.hstack(yVals))
        maxLims = [xMin, xMax, yMin - 0.05*abs(yMax - yMin), yMax + 0.05*abs(yMax - yMin)]
    elif globalPlotType[0] in [2]:
        maxLims = [np.amin(np.hstack(xVal)),np.amax(np.hstack(xVal)), None, None]
    return maxLims

def labels(nums, currentData, labs = None):
    global globalLegend

    if labs == None:
        labs = [globalLegend[currentData][0][ik] if type(ik) == int else globalLegend[ik[0]][0][ik[1]] for i, ik in enumerate(nums)]
        if len(nums) > 2:
            P.xlabel(labs[-1])
        else:
            P.xlabel(labs[-1])
            P.ylabel(labs[0])
    else:
        P.xlabel(labs[0])
        P.ylabel(labs[1])

def page():
    sns.set_palette(palettes[globalPageSetup['Palette']].values()[0])
    for key in globalPageSetup:
        try:
            mpl.rcParams['%s' % key] = globalPageSetup[key]
        except KeyError:
            pass

    fig = P.figure(1)
    ax = P.gca()

    ax.ticklabel_format(style='sci', scilimits=(0,0))

    ticker = mpl.ticker.ScalarFormatter(useMathText=True, useOffset=False)
    
    ax.xaxis.set_major_formatter(ticker)
    ax.xaxis.get_major_formatter().set_powerlimits((0,1))

    ax.yaxis.set_major_formatter(ticker)
    ax.yaxis.get_major_formatter().set_powerlimits((0,1))


    return (ax, fig)

def legend(inputNames, numbers, yVals, xVal, currentData):
    global globalLines
    ax = P.gca()
    names = []
    m = 0
    for key in numbers:
        for i in numbers[key]:
            for j in i[0]:
                globalLines[m].set_label(inputNames[key][0][j])
                names.append(inputNames[key][1][j])
                m += 1

    if globalLegendSetting == 2:
        labels = [ik.get_label() for i, ik in enumerate(globalLines)]
        P.legend(labels)
    if globalLegendSetting == 3:
        lineLabels(names, numbers, xVal, yVals)
    return

def lineLabels(names, numbers, xVal, yVals):
    textPlace = float(max(P.ylim())-min(P.ylim()))/50.
    #maxLimIndex = findNearest(xVal,max(P.xlim()))

    maxLimIndex = []
    for i in range(len(xVal)):
        maxLimIndex.append(findNearest(xVal[i],max(P.xlim())))

    #axpos = [[yVals[ik-numbers[-1]-1][maxLimIndex] for i, ik in enumerate(numbers[:-1])]]
    axpos = [[yVals[i][ik] for i, ik in enumerate(maxLimIndex)]]
    #axpos.append([names[1][ik-numbers[-1]-1] for i, ik in enumerate(numbers[:-1])])
    axpos.append([names[i] for i, ik in enumerate(maxLimIndex)])

    while True:
        for i in range(len(axpos[0])-1):
            if abs(axpos[0][i] - axpos[0][i+1]) < textPlace:
                axpos[0][i] -= textPlace/2.
                axpos[0][i+1] += textPlace/2.
                break
        else:
            break

    for i in range(len(axpos[0])):
        #P.text((max(P.xlim())-min(P.xlim()))*0.01 + max(P.xlim()), axpos[0][i], axpos[1][i])
        P.text(1.01, (axpos[0][i]-min(P.ylim()))/(max(P.ylim())-min(P.ylim())), axpos[1][i], transform=P.gca().transAxes)

def lineLabelShift(ax):
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()
    xVal = []
    yVals = []

    for l in ax.lines:
        xVal = l.get_data()[0]
        yVals.append(l.get_data()[1])

    textPlace = float(max(yLim)-min(yLim))/50.
    maxLimIndex = findNearest(xVal,max(xLim))

    for i, ik in enumerate(ax.lines):
        P.gca().texts[-(i+1)].set_y((yVals[i][maxLimIndex]-min(yLim))/(max(yLim)-min(yLim)))

def inset(ax, zoom, lims, xdata, ydata, locat = 1):
    axins = zoomed_inset_axes(ax, zoom, loc = locat)
    for i in range(len(ydata)):
        axins.plot(xdata[i],ydata[i])
    axins.set_xlim(lims[0], lims[1])
    axins.set_ylim(lims[2], lims[3])
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")


#==============================================================================
# Options, option menus and interpreting option commands
#==============================================================================

def drawMenu(items, menuName, controlFunc=None, *args):
    print ''
    for i, ik in enumerate(items):
        print '     %i) ' % (int(i)+1), ik

    try:
        num = int(raw_input('Choose %s option : ' % menuName))
        if controlFunc != None:
            returned = controlFunc(num, *args)
            return returned
        else:
            return num
    except ValueError:
        return


def Options(key, currentData, termWidth, vals):
    os.system('cls' if os.name == 'nt' else 'clear')

    if key in 'lL':
        title('Limit Options', termWidth)
        drawMenu(['Set limits', 'Clear limits'], 'limit', interpretLimitCommand)

    elif key in 'aA':
        title('Label Options', termWidth)
        drawMenu(['Set labels'], 'label', interpretLabelCommand)

    elif key in 'gG':
        title('Legend Options', termWidth)
        drawMenu(['Change legend titles', 'Change line labels', 'Set Legend Type', 'Line colours/styles'], 'legend', interpretLegendCommand, currentData)

    elif key in 'pP':
        title('Page Options', termWidth)
        drawMenu(['Page Size', 'Inset Axes', 'Smooth', 'Page Styles', 'Plot type'], 'page', interpretPageCommand)

    elif key in 'dD':
        title('Data Options', termWidth)
        vals = drawMenu(['Multiply Data', 'Sort Data', 'Read Units'], 'data', interpretDataCommand, currentData, vals)

    elif key in 'tT':
        title('Tunnel Options', termWidth)
        drawMenu(['Timescale'], 'tunnel', interpretTunnelCommand, vals)
    return vals
        
def interpretLimitCommand(option):
    global globalLims
    if option == 1:
        for i, ik in enumerate(globalLims):
            try:
                globalLims[i] = float(raw_input('Input %s %s-axis limit : ' % ('lower' if i == 0 or i == 2 else 'upper', 'x' if i < 2 else 'y')))
            except ValueError:
                globalLims[i] = None
        limits(globalLims)
        P.draw()
        return
    elif option == 2:
        globalLims = [None, None, None, None]
    else:
        return

def interpretLabelCommand(option):
    if option == 1:
        global globalLabs
        globalLabs[0] = defaultInput('Input x-axis label', globalLabs[0])
        globalLabs[1] = defaultInput('Input x-axis label', globalLabs[1])
    else:
        return

def interpretLegendCommand(option, currentData):
    global globalLegend
    if option == 1:
        renameCols = raw_input('Input the column numbers you wish to rename separated by a space : ').split()
        for i, ik in enumerate(renameCols):
            try:
                globalLegend[currentData][0][int(ik)-1] = raw_input('Please input new name for Column %s (currently labelled %s) : ' % (ik, globalLegend[currentData][0][int(ik)-1]))
            except ValueError:
                raw_input('The input %s is not an integer - Please press enter to return to main menu.' % ik)
                return
            except IndexError:
                raw_input('There is no Column %s - Please press enter to return to main menu.' % ik)
                return
    elif option == 2:
        renameCols = raw_input('Input the column numbers separated by a space for which you wish to change the line labels : ').split()
        for i, ik in enumerate(renameCols):
            try:
                globalLegend[currentData][1][int(ik)-1] = raw_input('Please input new line label for Column %s (currently labelled %s) : ' % (ik, globalLegend[currentData][1][int(ik)-1]))
            except ValueError:
                raw_input('The input %s is not an integer - Please press enter to return to main menu.' % ik)
                return
            except IndexError:
                raw_input('There is no Column %s - Please press enter to return to main menu.' % ik)
                return
    elif option == 3:
        global globalLegendSetting
        globalLegendSetting = drawMenu(['No Legend', 'Standard Legend', 'Line Labels'], 'legend type')

    elif option == 4:
        renameCols = raw_input('Input the column numbers you wish to recolour separated by a space : ').split()
        print 'My bad, this has not actually been implemented yet...'
    else:
        return

def interpretPageCommand(option):
    global globalPageSetup
    if option == 1:
        globalPageSetup['figure.figsize'][0] = float(defaultInput('Input x-value for page size', globalPageSetup['figure.figsize'][0]))
        globalPageSetup['figure.figsize'][1] = float(defaultInput('Input y-value for page size', globalPageSetup['figure.figsize'][1]))

    elif option == 2:
        global globalInsetToggle
        changeInsetToggle = raw_input('Would you like to change legend from %s to %s (default = yes): ' % ('On' if globalInsetToggle == True else 'Off', 'Off' if globalInsetToggle == True else 'On'))
        if changeInsetToggle in ['y', 'Y', 'yes', 'Yes', '']:
            globalInsetToggle = not globalInsetToggle
        else:
            pass
        
        if globalInsetToggle == True:
            global globalInsetLims
            globalInsetLims[4] = float(defaultInput('Input zoom level of inset', globalInsetLims[4]))
            globalInsetLims[0] = float(defaultInput('Input lower x-limit of inset', globalInsetLims[0]))
            globalInsetLims[1] = float(defaultInput('Input upper x-limit of inset', globalInsetLims[1]))
            globalInsetLims[2] = float(defaultInput('Input lower y-limit of inset', globalInsetLims[2]))
            globalInsetLims[3] = float(defaultInput('Input upper y-limit of inset', globalInsetLims[3]))

    elif option == 3:
        global smoothed
        smoothed = not smoothed

    elif option == 4:
        iPageSetup = drawMenu([ik['StyleName'] for i,ik in enumerate(styles)], 'style')
        globalPageSetup = styles[iPageSetup-1]

    elif option == 5:
        global globalPlotType

        globalPlotType[0] = drawMenu(['Line plot', 'Histogram', 'Scatter'], 'plot type')

        if globalPlotType[0] ==  2:
            globalPlotType[1] = drawMenu(['Histogram', 'KDE Histogram - Gaussian', 'KDE Histogram - Tophat', 'KDE Histogram - Exponential'], 'histogram type')
    return

def interpretDataCommand(option, currentData, vals):
    global dataMult
    if option == 1:
        cols = raw_input('Input the column numbers you wish to multiply separated by a space : ').split()
        cols = [int(ik) for i, ik in enumerate(cols)] 
        for i, ik in enumerate(cols):
            dataMult[currentData][ik-1] = float(defaultInput('Input multiplier to Column %i of Data %i' %(ik, currentData + 1), dataMult[currentData][i]))
            if yesNoSelect('Would you like to apply the same multiplier to the same column in all other datasets'):
                dataMult[:,ik-1] = np.ones(len(dataMult))*dataMult[currentData][ik-1]

    elif option == 2:
        col = defaultInput('Input the column number you wish to use to sort data', None, 'int')
        if col != None:
            col -= 1
            sorted_lists = sorted(izip(*vals[currentData]), key=lambda x: x[col])
            vals[currentData] = [[x[i] for x in sorted_lists] for i in range(len(vals[currentData]))]

    elif option == 3:
        units = readUnits(defaultInput('Input the path of the units file', None, 'string'))
        print 'Units available'
        for key in units:
            print str(key), ' : ', units[key]

        cols = raw_input('Input the column numbers you wish to multiply separated by a space : ').split()
        cols = [int(ik) for i, ik in enumerate(cols)]
        for i, ik in enumerate(cols):
            dataMult[currentData][ik-1] = float(units[defaultInput('Input name of unit multiplier for Column %i of Data %i' %(ik, currentData + 1), units.keys()[0], 'string')])
            if yesNoSelect('Would you like to apply the same multiplier to the same column in all other datasets'):
                dataMult[:,ik-1] = np.ones(len(dataMult))*dataMult[currentData][ik-1]
    return vals

def interpretTunnelCommand(option, vals):
    global globalLines

    if option == 1:
        print 'This tunnel should be used specifically for orbital separation evolution in common envelope systems.'
        print 'It will automatically pick out the different phases of evolution.'

        xData = np.array(vals[currentData][defaultInput('Input the column number for x-axis', 1, 'int') - 1])
        yData = np.array(vals[currentData][defaultInput('Input the column number for y-axis', 2, 'int') - 1])

        yDataAvg = np.copy(yData)
        gradient = []
        
        avgRange = 20


        for i,ik in enumerate(yData):
            if avgRange <= i < len(yData) - avgRange:
                yDataAvg[i] = np.mean(yData[i-avgRange:i+avgRange])
            elif i < avgRange:
                yDataAvg[i] = np.mean(yData[0:i+avgRange])
            elif i >= len(yData) - avgRange:
                yDataAvg[i] = np.mean(yData[i-avgRange:-1])

        for i,ik in enumerate(yData):
            if avgRange <= i < len(yData) - avgRange:
                gradient.append((yDataAvg[i-avgRange] - yDataAvg[i+avgRange]) / float(2*avgRange))
            elif i >= len(yData) - avgRange:
                gradient.insert(-1,(yDataAvg[i-avgRange] - yDataAvg[-1]) / float(len(yDataAvg)-i+avgRange))

        for i in range(len(yData),-1,-1):
            if i < avgRange:
                gradient.insert(0,(yDataAvg[0] - yDataAvg[i+avgRange]) / float(i+avgRange))

        command = [1,0]

        xData, yData, yDataAvg, gradient = [xData], [yData], [yDataAvg], np.array([gradient])

        makePlot(command, currentData, [1,1], xData, gradient)
        P.show()
    else:
        pass
    return

#==============================================================================
# To do with User Interface
#==============================================================================


def interpretNumString(inString):
    rangeExp = re.compile(r'((\d+)-(\d+))')
    outString = inString
    for m in rangeExp.finditer(inString):
        outString = re.sub(m.group(1), ' '.join(map(str,range(int(m.group(2)),int(m.group(3))+1) if int(m.group(2)) < int(m.group(3)) else range(int(m.group(2)),int(m.group(3))-1, -1))), outString)
    outString = map(lambda x: int(x) - 1,re.sub(r',',r' ',outString).split())
    return outString

def interpretUICommand(line, vals, currentData, termWidth = 80):
    global globalLims, globalLabs, globalPlotType, globalLines
    running = True
    command = None
    try:
        cols = re.findall(r'(\d?){([\d\-\s:,]+)}', line, re.I)
        if cols == []:
            if re.match(r'[\d ]', line, re.I) is not None:
                cols = ':'.join(line.rsplit(' ', 1))
                cols = [(str(currentData),cols.replace(' ', ','))]
            elif re.match(r'[qnvlagpdt]', line, re.I) is not None:
                command = re.match(r'([qnvlagpdt])', line, re.I).group(0)

        dataOrder = range(len(vals))
        dataOrder = dataOrder[currentData:] + dataOrder[:currentData]
        plotCols = {(dataOrder[i] if ik[0] == '' else int(ik[0])) : ik[-1].split() for i,ik in enumerate(cols)}
        for key in plotCols:
            plotCols[key] = [ik.split(':') for i,ik in enumerate(plotCols[key])]

        for key in plotCols:
            plotCols[key] = [[interpretNumString(plotCols[key][j][i]) for i,ik in enumerate(plotCols[key][j])] for j,jk in enumerate(plotCols[key])]

        yVals = []
        xVal = []

        if (globalPlotType[0] in [1,3]) and (plotCols != {}):
            for key in plotCols:
                for i in plotCols[key]:
                    for j in i[0]:
                        
                        yVals.append([float(k) for k in vals[key][j]])
                        xVal.append([float(k) for k in vals[key][i[1][0]]])

            xVal = np.array(xVal)
            yVals = np.array(yVals)

            for key in plotCols:
                for i,ik in enumerate(plotCols[key]):
                    for j in ik[0]:
                        xVal[i] = [k * dataMult[key][ik[1][0]] for k in xVal[i]]
                        yVals[i] = [k * dataMult[key][j-1] for k in yVals[i]]

        elif (globalPlotType[0] in [2]) and (plotCols != {}):
            for key in plotCols:
                xVal.append([float(j) for j in vals[key][plotCols[key][0][0][0]]])

            xVal = np.array(xVal)

            for i,ik in enumerate(plotCols.keys()):
                xVal[i] = xVal[i]*dataMult[i][ik]

        
        if plotCols != {}:
            makePlot(plotCols, currentData, globalPlotType, xVal, yVals)
        else:
            if command is None:
                pass
            elif command in 'qQ':
                running = False
            elif command in 'nN':
                if currentData < len(vals) - 1:
                    currentData += 1
                else:
                    currentData = 0
            elif command in 'vV':
                if currentData > 0:
                    currentData -= 1
                else:
                    currentData = len(vals) - 1
            elif command in 'lLaAgGpPdDtT':
                vals = Options(command, currentData, termWidth, vals)
    except ValueError:
        print '\nPlease format plot requests as \"y y y (...) x\", i.e. \"2 1\" plots Column 2 against Column 1.\n'
        raw_input('Press Enter key to continue...')
        pass
    except IndexError:
        print '\nPlease format plot requests as \"y y y (...) x\", i.e. \"2 1\" plots Column 2 against Column 1.\n'
        raw_input('Press Enter key to continue...')
        pass
    return (running, currentData)

def drawUI(termWidth, vals, currentData, legend):
    os.system('cls' if os.name == 'nt' else 'clear')
    numCols = int(floor(float(termWidth)/30.))
    colWidth = int(floor(float(termWidth)/float(numCols)))

    length = int(floor(len(legend)/numCols)*numCols)
    if len(legend)%numCols != 0:
        length += numCols

    trueNumCols = int(ceil(float(len(legend))/float(length) * numCols))
    sideBuffer = (termWidth - (trueNumCols * colWidth) ) / 2

    maxTextWidth = [min([(colWidth + max([len(ik) for i,ik in enumerate(globalLegend[currentData][0][j*length/trueNumCols:(j+1)*length/trueNumCols])]) + len(str(len(globalLegend[currentData]))) + 2)/2, colWidth]) for j in range(trueNumCols)]

    title('Welcome to Pyro!', termWidth)
    title('Data Set %i' % (currentData + 1), termWidth, 2)
    for i in range(length/numCols):
        sys.stdout.write(' ' * sideBuffer)
        for j in range(numCols):
            try:
                sys.stdout.write(('{:>%i}'%colWidth).format(('{:<%i.%i}'%(maxTextWidth[j],colWidth)).format(' %i. ' % (1 + i + j * length / numCols) + legend[i + j * length / numCols]))),
            except IndexError:
                sys.stdout.write('')
        print ''

    options = ['[L]imits', 'L[a]bels', 'Le[g]end', '[P]age', '[D]ata', '[N]ext Data', 'Pre[v]ious Data', '[T]unnels']
    print '-'*termWidth
    for i in range(int(ceil(len(options)/4.))):
        sys.stdout.write('-'*( (termWidth - 4 * 15) / 2 - 1) + '|'),
        for j in range(4):
            try:
                sys.stdout.write("{:^15}".format(options[(i * 4) + j])),
            except IndexError:
                sys.stdout.write(''.ljust(15))
        sys.stdout.write('|' + '-'*( int(ceil( (termWidth - 4 * 15) / 2.)) - 1))
    print '-'*termWidth

    command = raw_input('Select from above options : ')
    return command

def title(message, termWidth, titleType = 1):
    if titleType == 2:
        sys.stdout.write(("{:^%i}"%termWidth).format(message))
        sys.stdout.write('-'*termWidth)
    else:
        sys.stdout.write('-'*termWidth)
        sys.stdout.write(("{:-^%i}"%termWidth).format('|' + ' '*3 + message + ' '*3 + '|'))
        sys.stdout.write('-'*termWidth)


#==============================================================================
# Running program and setting initial conditions
#==============================================================================

running = True
currentData = 0
vals,globalLegend = readFile()
styles = readStyle()
palettes = readPalette()

if len(styles) == 0:
    globalPageSetup = {'lines.linewidth' : 1.5, 'figure.facecolor' : 'w', 'figure.figsize' : [12, 8]}
else:
    globalPageSetup = styles[0]

if len(palettes) == 0:
    palettes = [{'Name 1' : sns.color_palette(["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD", "#707070", "#753435", "#80523B", "#584581"])}]
    globalPageSetup['Palette'] = 0

dataMult = {i : np.ones(len(vals[i])) for i,ik in enumerate(vals)}

for j in range(len(vals)):
    globalLegend[j].append(['Col. %i' %(i+1) for i, ik in enumerate(vals[j])])
globalInsetToggle = False
globalLegendSetting = 2
#palettes = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind', 'Paired', 'hls', 'husl', 'BuGn', 'GnBu', 'OrRd', 'PuBu', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'BrBG', 'PiYG', 'PRGn', 'PuOr', 'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral']
globalLims = [None, None, None, None]
globalInsetLims = [0.0, 0.0, 0.0, 0.0, 2.0]
globalLabs = [None,None]

globalLines = []
globalPlotType = [1, 1]
isPicked = False
smoothed = False

while running == True:
    termWidth = getWinWidth()
    command = drawUI(termWidth, vals, currentData, globalLegend[currentData][0])
    running, currentData = interpretUICommand(command, vals, currentData, termWidth)
