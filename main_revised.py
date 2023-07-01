import codecs
import operator
from flask import Flask, make_response, request, Response, render_template, send_file
import csv
import numpy as np
from scipy.stats import norm
from scipy.stats import rankdata
from scipy.stats import chi
import io
import matplotlib.backends
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import base64

app = Flask(__name__)

SMALL_SIZE = 36
MEDIUM_SIZE = 48
BIGGER_SIZE = 72


def geo_mean(iterable):  # calculate a geometric mean
    a = np.array(iterable)
    return a.prod() ** (1.0 / len(a))


plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

font = {'family': 'sans-serif',
        'color': 'black',
        'weight': 'normal',
        'size': 16,
        }


def variance(prodPercentiles, suscPercentiles):  # output associated variances related to underlying analysis
    """find common denominator to calculate multipliers for variance calculations"""
    tempProd = max([1 / eval(prodPercentiles[0]), 1 / (eval(prodPercentiles[1]) - eval(prodPercentiles[0])),
                    1 / (1 - eval(prodPercentiles[1]))])
    tempSusc = max([1 / eval(suscPercentiles[0]), 1 / (eval(suscPercentiles[1]) - eval(suscPercentiles[0])),
                    1 / (1 - eval(suscPercentiles[1]))])
    tempBoth = [tempProd, tempSusc]
    permBoth = [0, 0]
    counted = [0, 0]
    for i in range(2):
        counting = 1
        flag = True
        while flag == True:
            integerCheck = tempBoth[i] * counting
            counting += 1
            if integerCheck % 1 == 0:
                flag = False
                permBoth[i] = integerCheck
                counted[i] = counting

    multipliersProd = [permBoth[0] / eval(prodPercentiles[0]),
                       permBoth[0] / (eval(prodPercentiles[1]) - eval(prodPercentiles[0])),
                       permBoth[0] / (1 - eval(prodPercentiles[1]))]
    multipliersSusc = [permBoth[1] / eval(suscPercentiles[0]),
                       permBoth[1] / (eval(suscPercentiles[1]) - eval(suscPercentiles[0])),
                       permBoth[1] / (1 - eval(suscPercentiles[1]))]

    addVarianceProd = (multipliersProd[0] + multipliersProd[2]) / sum(multipliersProd)
    addVarianceSusc = (multipliersSusc[0] + multipliersSusc[2]) / sum(multipliersSusc)

    multMean = np.log(6) / 3
    multVarianceProd = (multipliersProd[0] * (multMean) ** 2 + multipliersProd[1] * (multMean - np.log(2)) ** 2 +
                        multipliersProd[2] * (multMean - np.log(3)) ** 2) / sum(multipliersProd)
    multVarianceSusc = (multipliersSusc[0] * (multMean) ** 2 + multipliersSusc[1] * (multMean - np.log(2)) ** 2 +
                        multipliersSusc[2] * (multMean - np.log(3)) ** 2) / sum(multipliersSusc)

    return [addVarianceProd, addVarianceSusc, multVarianceProd, multVarianceSusc]


def addAnalysis(data, num, var, type, reverse):  # additive model output scores and associated standard errors
    error = []
    scores = []
    for i in range(len(data)):
        prescore = []
        weight = []
        for j in range(1, 2 + num):
            if data[i][j] != '':
                weight.append(float(data[i][j + num + 2]))
                for b in range(int(data[i][j + num + 2])):
                    if type == 'prod' and reverse in ['y', 'Y']:
                        prescore.append(4 - float(data[i][j]))
                    else:
                        prescore.append(float(data[i][j]))
        if prescore != []:
            scored = np.average(prescore)
            scores.append(scored)
            totalWeight = np.sum(weight)

            for z in range(len(weight)):
                weight[z] = (weight[z] / totalWeight) ** 2
            error.append((var * np.sum(weight)) ** (1 / 2))
    return scores, error


def multAnalysis(data, num, var, type, reverse):  # multiplicative model output scores and associated standard errors
    error = []
    scores = []
    logscores = []
    for i in range(len(data)):
        prescore = []
        weight = []
        for j in range(1, 2 + num):
            if data[i][j] != '':
                weight.append(float(data[i][j + num + 2]))
                for b in range(int(data[i][j + num + 2])):
                    if type == 'prod' and reverse in ['y', 'Y']:
                        prescore.append(4 - float(data[i][j]))
                    else:
                        prescore.append(float(data[i][j]))
        if prescore != []:
            totalWeight = np.sum(weight)

            scored = geo_mean(prescore)
            logscores.append(np.log(scored))
            scores.append(scored)

            for z in range(len(weight)):
                weight[z] = (weight[z] / totalWeight) ** 2
            error.append((var * np.sum(weight)) ** (1 / 2))
    return scores, error, logscores


def inputData(data, data2):
    prodScaling = data2[0][1]  # store scaling choice (multiplicative or additive) for productivity attributes
    suscScaling = data[0][1]  # store scaling choice (multiplicative or additive) for susceptibility attributes
    scaling = [prodScaling, suscScaling]
    prodPercentiles = data2[1][
                      1:3]  # store percentile cut-offs for attribute scores (1,2,3) for productivity attributes
    suscPercentiles = data[1][
                      1:3]  # store percentile cut-offs for attribute scores (1,2,3) for susceptibility attributes
    prodThresholds = data2[1][1:3]  # store productivity thresholds
    suscThresholds = data[1][1:3]  # store susceptibility thresholds
    nullVal = data[0][4]
    reverse = data2[0][4]
    xaxis = data2[1][4]
    yaxis = data[1][4]
    weightS = float(data[2][4])
    axisLabel = [xaxis, yaxis]
    if prodThresholds != suscThresholds:
        print('Warning: Thresholds in Productivity spreadsheet do not match thresholds in Susceptibility spreadsheet!')

    varianceList = variance(prodPercentiles, suscPercentiles)

    numProd = int(data2[2][1])  # store number of productivity attributes used in analysis
    numSusc = int(data[2][1])  # store number of susceptibility attributes used in analysis

    data = data[6:]  # exclude first row (data header)
    data2 = data2[6:]

    newdata = []
    newdata2 = []
    species = []

    for x in range(len(data)):  # exclude first column
        newdata.append(data[x][1:])
        newdata2.append(data2[x][1:])
        if data2[x][1] != '':
            species.append(data2[x][1])

    mean = [2, 2]

    """define scaling model based on first letter in scaling input"""

    if scaling[0][0] == 'a' or scaling[0][0] == 'A':
        productivity, producError = addAnalysis(newdata2, numProd, varianceList[0], 'prod', reverse)
        mean[0] = 2
        choiceProd = productivity
    else:
        productivity, producError, logproductivity = multAnalysis(newdata2, numProd, varianceList[2], 'prod', reverse)
        mean[0] = np.log(6) / 3
        choiceProd = logproductivity

    if scaling[1][0] == 'a' or scaling[1][0] == 'A':
        susceptibility, susceptError = addAnalysis(newdata, numSusc, varianceList[1], 'susc', reverse)
        mean[1] = 2
        choiceSus = susceptibility
    else:
        susceptibility, susceptError, logsusceptibility = multAnalysis(newdata, numSusc, varianceList[3], 'susc',
                                                                       reverse)
        mean[1] = np.log(6) / 3
        choiceSus = logsusceptibility
    return producError, susceptError, mean, prodThresholds, choiceProd, choiceSus, productivity, susceptibility, species, nullVal, axisLabel, weightS


def secondary(producError, susceptError, mean, prodThresholds, choiceProd, choiceSus, productivity, susceptibility,
              species, nullVal, axisLabel, weightS):
    SEp = producError
    SEs = susceptError
    SEps = []
    riskVector = []
    projectionMatrix = []
    meanT = np.matrix(mean).transpose()
    transformation = []
    projection = []
    projection_p = []
    projection_s = []
    distanceMetric = []
    revisedCategory = []
    low = 0
    medium = 0
    high = 0

    lowerThresh = norm.ppf(eval(prodThresholds[0]))
    upperThresh = norm.ppf(eval(prodThresholds[1]))
    # print(lowerThresh, upperThresh)
    newVuln = []

    """calculate vulnerabilities based on projections to risk axis and assign categories low, medium, high"""
    for i in range(len(SEp)):
        SEps.append(np.sqrt(2) * SEp[i] * SEs[i] / np.sqrt(SEp[i] ** 2 + SEs[i] ** 2))

        riskVector.append(np.matrix([SEs[i], SEp[i]]).transpose())
        projectionMatrix.append(
            riskVector[i] * (riskVector[i].transpose() * riskVector[i]) ** -1 * riskVector[i].transpose())

        transformation.append(meanT - projectionMatrix[i] * meanT)
        projection.append(projectionMatrix[i] * np.matrix([choiceProd[i], choiceSus[i]]).transpose())
        projection_p.append(projection[i][0, 0] + transformation[i][0, 0])
        projection_s.append(projection[i][1, 0] + transformation[i][1, 0])
        diffp = projection_p[i] - meanT[0, 0]
        diffs = projection_s[i] - meanT[1, 0]
        distanceMetric.append(np.sign(diffp) * np.sqrt((diffp) ** 2 + (diffs) ** 2))
        print(distanceMetric[i])
        newVuln.append(norm.cdf(distanceMetric[i] / SEps[i]))
        if distanceMetric[i] < lowerThresh * SEps[i]:
            low += 1
            revisedCategory.append('low')
        elif lowerThresh * SEps[i] <= distanceMetric[i] < upperThresh * SEps[i]:
            medium += 1
            revisedCategory.append('medium')
        else:
            high += 1
            revisedCategory.append('high')

    markerColor = []
    for w in range(len(revisedCategory)):
        if revisedCategory[w] == 'low':
            markerColor.append('b')
        elif revisedCategory[w] == 'medium':
            markerColor.append('y')
        else:
            markerColor.append('r')

    markerArea = []
    for i in range(len(revisedCategory)):
        count = 0
        uniqueMeasure = [productivity[i], susceptibility[i]]
        for x in range(len(revisedCategory)):
            if [productivity[x], susceptibility[x]] == uniqueMeasure:
                count += 1
        markerArea.append(200 * count)

    return species, newVuln, revisedCategory, productivity, susceptibility, markerColor, markerArea, nullVal, SEs, SEp, mean, prodThresholds, choiceProd, choiceSus, axisLabel, weightS


def main(data, data2):

    producError, susceptError, mean, prodThresholds, choiceProd, choiceSus, productivity, susceptibility, species, nullVal, axisLabel, weightS = inputData(
        data, data2)

    return secondary(producError, susceptError, mean, prodThresholds, choiceProd, choiceSus, productivity,
                     susceptibility, species, nullVal, axisLabel, weightS)


def plot(result, color, area):
    img = io.BytesIO()
    plt.figure(figsize=(10, 10))
    plt.scatter(result[0], result[1], c=color, s=area, alpha=0.4)
    plt.axis((1, 3, 1, 3))
    plt.xlabel(result[2][0])
    plt.ylabel(result[2][1])
    plt.tight_layout()
    plt.xticks([1, 2, 3])
    plt.yticks([1, 2, 3])
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plotUrl = base64.b64encode(img.getvalue()).decode('cp1252', errors='ignore')
    return plotUrl


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/display', methods=["POST"])
def transform_view():
    request_files = request.files.getlist('data_file')
    if not request_files:
        return "No file"

    stream = codecs.iterdecode(request_files[0].stream, 'cp1252', errors='ignore')
    zstream = codecs.iterdecode(request_files[-1].stream, 'cp1252', errors='ignore')
    yStream = []
    spanList = []
    for r in range(1, len(request_files) - 1):
        yStream.append(list(csv.reader(codecs.iterdecode(request_files[r].stream, 'cp1252', errors='ignore'))))
        spanList.append(r - 1)

    stream = list(csv.reader(stream))
    zstream = list(csv.reader(zstream))

    plot_url = []
    results = []
    numer = []

    nullMean = []
    fullMean = []
    fullSusc = []
    fullErrS = []
    fullWeights = []

    for s in range(len(yStream)):
        species, vuln, category, produc, susc, color, area, nullVal, SEs, SEp, mean, prodThresholds, choiceProd, choiceSus, axisLabel, weightS = main(
            yStream[s], stream)
        result = []
        fullWeights.append(weightS)
        for i in range(len(species)):
            result.append([species[i], round(produc[i], 1), round(susc[i], 1), round(vuln[i], 2), category[i]])

        results.append(result)
        toPlot = [produc, susc, axisLabel]
        fullSusc.append(choiceSus)
        fullErrS.append(SEs)
        fullMean.append(mean[1])

        numer.append([])

        if nullVal == 'n' or nullVal == 'N':
            nullMean.append(mean[1])
        elif nullVal == 'a' or nullVal == 'A':
            if mean[1] == np.log(6) / 3:
                nullMean.append(1.098)
            else:
                nullMean.append(3)
        else:
            if mean[1] == np.log(6) / 3:
                nullMean.append(0)
            else:
                nullMean.append(1)


        for t in range(len(species)):
            numer[s].append(SEs[t] * weightS / mean[1])

        plot_url.append(plot(toPlot, color, area))

    sError = []
    sVal = []
    sPlot = []
    sumWeights = sum(fullWeights)

    corrMatrix = np.array(zstream[0:len(yStream)][0:len(yStream)]).astype(float)

    for u in range(len(species)):
        tempM = []
        tempN = []
        tempS = []
        plotS = []
        for v in range(len(numer)):
            tempM.append(fullSusc[v][u] / fullMean[v] * fullWeights[v])
            tempN.append(nullMean[v] / fullMean[v] * fullWeights[v])
            if fullMean[v] == np.log(6) / 3:
                tempS.append(np.exp(fullSusc[v][u]) * fullWeights[v] / sumWeights)
                plotS.append(np.exp(nullMean[v]) * fullWeights[v] / sumWeights)
            else:
                tempS.append(fullSusc[v][u] * fullWeights[v] / sumWeights)
                plotS.append(nullMean[v] * fullWeights[v] / sumWeights)

        cov = np.sum(np.matmul(np.matmul(np.array([item[u] for item in numer]), corrMatrix),
                               np.transpose(np.array([item[u] for item in numer]))))

        SE = np.sqrt(cov)
        tempVal = sum(tempM)
        sError.append(SE)

        sVal.append(tempVal)
        plotVal = norm.cdf(sum(tempS), loc=sum(plotS), scale=SE)
        plotVal2 = norm.ppf(plotVal, loc=2, scale=SE * 2 / sum(plotS))
        plotVal3 = min(plotVal2, 3)
        plotVal4 = max(plotVal3, 1)
        sPlot.append(plotVal4)


    fMean = [mean[0], sum(tempN)]

    speciesAll, vulnAll, categoryAll, producAll, suscAll, colorAll, areaAll, nullVal, SEs, SEp, mean, prodThresholds, choiceProd, choiceSus, axisLabel, weightS = secondary(
        SEp, sError, fMean, prodThresholds, choiceProd, sVal, produc, sPlot, species, nullVal, axisLabel, weightS)
    fullResults = []
    for i in range(len(species)):
        fullResults.append(
            [speciesAll[i], round(producAll[i], 1), round(suscAll[i], 1), round(vulnAll[i], 2), categoryAll[i]])

    toPlot2 = [producAll, suscAll, axisLabel]

    plot_url2 = plot(toPlot2, colorAll, areaAll)

    return render_template('display.html', my_list=results, plot_url=plot_url, spanList=spanList, my_list2=fullResults,
                           plot_url2=plot_url2, label=axisLabel)


if __name__ == "__main__":
    app.run(host="localhost", port=8080, debug=True)