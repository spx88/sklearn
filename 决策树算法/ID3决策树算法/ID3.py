import math
import operator
import treePlotter


# 年龄 青年 0 中年1 老年 2
# 信贷 一般 0 好1 非常好2
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['age', 'work', 'house', 'credit', 'apply']
    # change to discrete values
    return dataSet, labels


# 计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:  # the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * math.log(prob, 2)  # log base 2
    return shannonEnt


# 按特征和特征值划分数据集，数据一定是列表，最后一列是类别，axis是位置，value是匹配的值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)  # 这里注意extend和append的区别
    return retDataSet


# ID3决策树分类算法
def chooseBestFeatureToSplitID3(dataSet):
    numFeatures = len(dataSet[0]) - 1  # myDat[0]表示第一行数据, 最后一列用作标签,这里相当于计算特征数为4
    baseEntropy = calcShannonEnt(dataSet)  # 计算原始数据的熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 遍历所有的特征
        featList = [example[i] for example in dataSet]  # featList是每一列的所有值，是一个列表
        uniqueVals = set(featList)  # 集合中每个值互不相同
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy

        print('信息增益', infoGain)
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # returns an integer


# 当所有特征都用完的时候投票决定分类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# ID3构建树
def createTreeID3(dataSet, labels):
    classList = [example[-1] for example in dataSet]  # 创建类标签
    if classList.count(classList[0]) == len(classList):  # 终止条件1 如果所有类标签完全相同则停止，直接返回该类标签
        return classList[0]
    if len(dataSet[0]) == 1:  # 终止条件2 遍历完了所有的标签仍然不能将数据集划分为仅包含唯一类别的分组
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplitID3(dataSet)  # 最佳分类特征的下标
    bestFeatLabel = labels[bestFeat]  # 最佳分类特征的名字
    print('最佳分类特征', bestFeatLabel)
    myTree = {bestFeatLabel: {}}  # 创建
    del (labels[bestFeat])  # 从label里删掉这个最佳特征，创建迭代的label列表
    featValues = [example[bestFeat] for example in dataSet]  # 这个最佳特征对应的所有值
    uniqueVals = set(featValues)
    for value in uniqueVals:  # 遍历最佳特征的所有特征值，连续递归划分子树
        subLabels = labels[:]  # 拷贝，防止构建子树时删除特征相互干扰
        myTree[bestFeatLabel][value] = createTreeID3(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == '__main__':
    myDat, labels = createDataSet()
    print(createTreeID3(myDat, labels))
