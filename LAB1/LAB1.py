"""
run this grogram: (default m=2)
python LAB1.py -m 2 (Apriori)
python LAB1.py -m 3 (FPgrowth)
"""
import os
import time
from tqdm import tqdm, trange
from itertools import chain, combinations, count
from collections import defaultdict
from optparse import OptionParser



# -------------- FPgrowth --------------
# cited from https://github.com/Ryuk17/MachineLearning
class FPNode:
    def __init__(self, item, count, parent):
        self.item = item
        # support
        self.count = count
        self.parent = parent
        # the same elements
        self.next = None
        self.children = {}

    def display(self, ind=1):
        print(''*ind, self.item, '', self.count)
        for child in self.children.values():
            child.display(ind+1)

class FPgrowth:
    def __init__(self, min_support, min_confidence=0):
        self.min_support = min_support
        self.min_confidence = min_confidence

    '''
    Function:  transfer2FrozenDataSet
    Description: transfer data to frozenset type
    Input:  data              dataType: ndarray     description: train_data
    Output: frozen_data       dataType: frozenset   description: train_data in frozenset type
    '''
    def transfer2FrozenDataSet(self, data):
        frozen_data = {}
        for elem in data:
            if frozenset(elem) in frozen_data.keys():
                frozen_data[frozenset(elem)] += 1
            else:
                frozen_data[frozenset(elem)] = 1
        return frozen_data

    '''
      Function:  updataTree
      Description: updata FP tree
      Input:  data              dataType: ndarray     description: ordered frequent items
              FP_tree           dataType: FPNode      description: FP tree
              header            dataType: dict        description: header pointer table
              count             dataType: count       description: the number of a record 
    '''
    def updataTree(self, data, FP_tree, header, count):
        frequent_item = data[0]
        if frequent_item in FP_tree.children:
            FP_tree.children[frequent_item].count += count
        else:
            FP_tree.children[frequent_item] = FPNode(frequent_item, count, FP_tree)
            if header[frequent_item][1] is None:
                header[frequent_item][1] = FP_tree.children[frequent_item]
            else:
                self.updateHeader(header[frequent_item][1], FP_tree.children[frequent_item]) # share the same path

        if len(data) > 1:
            self.updataTree(data[1::], FP_tree.children[frequent_item], header, count)  # recurrently update FP tree

    '''
      Function: updateHeader
      Description: update header, add tail_node to the current last node of frequent_item
      Input:  head_node           dataType: FPNode     description: first node in header
              tail_node           dataType: FPNode     description: node need to be added
    '''
    def updateHeader(self, head_node, tail_node):
        while head_node.next is not None:
            head_node = head_node.next
        head_node.next = tail_node

    '''
      Function:  createFPTree
      Description: create FP tree
      Input:  train_data        dataType: ndarray     description: features
      Output: FP_tree           dataType: FPNode      description: FP tree
              header            dataType: dict        description: header pointer table
    '''
    def createFPTree(self, train_data):
        initial_header = {}
        # 1. the first scan, get singleton set
        for record in train_data:
            for item in record:
                initial_header[item] = initial_header.get(item, 0) + train_data[record]

        # get singleton set whose support is large than min_support. If there is no set meeting the condition,  return none
        header = {}
        for k in initial_header.keys():
            if initial_header[k] >= self.min_support:
                header[k] = initial_header[k]
        frequent_set = set(header.keys())
        if len(frequent_set) == 0:
            return None, None

        # enlarge the value, add a pointer
        for k in header:
            header[k] = [header[k], None]

        # 2. the second scan, create FP tree
        # root node
        FP_tree = FPNode('root', 1, None)
        for record, count in train_data.items():
            frequent_item = {}
            for item in record:                # if item is a frequent set， add it
                if item in frequent_set:       # 2.1 filter infrequent_item
                    frequent_item[item] = header[item][0]

            if len(frequent_item) > 0:
                # 2.1 sort all the elements in descending order according to count
                ordered_frequent_item = [val[0] for val in sorted(frequent_item.items(), key=lambda val:val[1], reverse=True)]
                # 2.2 insert frequent_item in FP-Tree， share the path with the same prefix
                self.updataTree(ordered_frequent_item, FP_tree, header, count)

        return FP_tree, header

    '''
      Function: ascendTree
      Description: ascend tree from leaf node to root node according to path
      Input:  node           dataType: FPNode     description: leaf node
      Output: prefix_path    dataType: list       description: prefix path
              
    '''
    def ascendTree(self, node):
        prefix_path = []
        while node.parent != None and node.parent.item != 'root':
            node = node.parent
            prefix_path.append(node.item)
        return prefix_path

    '''
    Function: getPrefixPath
    Description: get prefix path
    Input:  base          dataType: FPNode     description: pattern base
            header        dataType: dict       description: header
    Output: prefix_path   dataType: dict       description: prefix_path
    '''
    def getPrefixPath(self, base, header):
        prefix_path = {}
        start_node = header[base][1]
        prefixs = self.ascendTree(start_node)
        if len(prefixs) != 0:
            prefix_path[frozenset(prefixs)] = start_node.count

        while start_node.next is not None:
            start_node = start_node.next
            prefixs = self.ascendTree(start_node)
            if len(prefixs) != 0:
                prefix_path[frozenset(prefixs)] = start_node.count
        return prefix_path

    '''
    Function: findFrequentItem
    Description: find frequent item
    Input:  header               dataType: dict       description: header [name : (count, pointer)]
            prefix               dataType: dict       description: prefix path
            frequent_set         dataType: set        description: frequent set
    '''
    def findFrequentItem(self, header, prefix, frequent_set):
        # for each item in header, then iterate until there is only one element in conditional fptree
        try:
            header_items = [val[0] for val in sorted(header.items(), key=lambda val: val[1][0])]
        except AttributeError:
            return
        if len(header_items) == 0:
            return

        for base in header_items:
            new_prefix = prefix.copy()
            new_prefix.add(base)
            support = header[base][0]
            frequent_set[frozenset(new_prefix)] = support

            prefix_path = self.getPrefixPath(base, header)
            if len(prefix_path) != 0:
                conditonal_tree, conditional_header = self.createFPTree(prefix_path)
                if conditional_header is not None:
                    self.findFrequentItem(conditional_header, new_prefix, frequent_set)

    '''
      Function:  train
      Description: train the model
      Input:  train_data       dataType: ndarray   description: items
              display          dataType: bool      description: print the rules
      Output: rules            dataType: list      description: the learned rules
              frequent_items   dataType: list      description: frequent items set
    '''
    def train(self, data):
        data = list(data)
        data_len = len(data)
        data = self.transfer2FrozenDataSet(data)
        self.min_support = self.min_support * (data_len)
        FP_tree, header = self.createFPTree(data)
        # FP_tree.display()
        frequent_set = {}
        prefix_path = set([])
        self.findFrequentItem(header, prefix_path, frequent_set)

        frequent_set.items()
        frequent_set_out = tuple()
        for key in frequent_set.keys():
            key_new = tuple(list(key))
            frequent_set_out += ((key_new, frequent_set[key] / float(data_len)),)

        return frequent_set_out



# -------------- Apriori --------------
def subsets(arr):
    """Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])

def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet, k):
    """calculates the support for items in the itemSet and returns a subset
    of the itemSet each of whose elements satisfies the minimum support"""
    _itemSet = set()
    localSet = defaultdict(int)
    
    for item in itemSet:
        for transaction in transactionList:
            if item.issubset(transaction):
                freqSet[item] += 1
                localSet[item] += 1

    for item, count in localSet.items():
        support = float(count) / len(transactionList)

        if support >= minSupport:
            _itemSet.add(item)

    return _itemSet

def joinSet(itemSet, length):
    """Join a set with itself and returns the n-element itemsets"""
    return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])

def getItemSetTransactionList(dataIterator):
    """get all 1-item itemset and all transactions"""
    transactionList = list()
    itemSet = set()
    for record in dataIterator:
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))  # Generate 1-itemSets
            
    return itemSet, transactionList

def pruning(candidateSet, prevFreqSet, length):
    tempCandidateSet = candidateSet.copy()
    for item in candidateSet:
        subsets = combinations(item, length)
        for subset in subsets:
            # if the subset is not in previous K-frequent get, then remove the set
            if(frozenset(subset) not in prevFreqSet):
                tempCandidateSet.remove(item)
                break
    return tempCandidateSet

def runApriori(dataIter, minSupport):
    # -------------- task1 --------------
    # record itemset, transaction
    itemSet, transactionList = getItemSetTransactionList(dataIter)

    # initialize
    freqSet = defaultdict(int)
    largeSet = dict()
    candidatesRatio = []
    k = 1

    # find all frequent itemsets
    # find 1-item itemset first
    currentLSet = returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet, k)
    candidatesRatio.append([k, len(currentLSet), len(currentLSet)])
    k = k + 1
    # find k-item itemset until no k-item itemset with support over minimun support
    while currentLSet != set([]):
        # store frequent itemset
        largeSet[k - 1] = currentLSet
        # generate new candidates
        candidateSet = joinSet(currentLSet, k)
        # perform subset testing and remove pruned supersets
        beforePrunNum = len(candidateSet)
        candidateSet = pruning(candidateSet, currentLSet, k-1)
        afterPrunNum = len(candidateSet)
        # scanning itemSet for counting support and discard failed itemsets
        currentLSet = returnItemsWithMinSupport(candidateSet, transactionList, minSupport, freqSet, k)
        candidatesRatio.append([k, beforePrunNum, afterPrunNum])
        k = k + 1

    # record all frequent itemsets with support
    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), (lambda item: float(freqSet[item]) / len(transactionList))(item)) for item in value])

    return candidatesRatio, toRetItems

# print results of Task1 and Task2
def printResultsFile1(items, minSupport, fname, mode):
    with open("step"+mode+"_task1_"+fname+"_"+minSupport*100+"%_result1.txt", 'w') as f:
        for item, support in sorted(items, key=lambda x: x[1], reverse=True):
            tmp = str("{0:.1f}%\t{1}\n".format(support*100, item)).replace("'", "").replace("(", "{").replace(")", "}")
            tmp = tmp[::-1].replace(",", "", 1)
            tmp = tmp[::-1]
            f.write(tmp)

def printResultsFile2(candidates, items_len, minSupport, fname, mode):
    with open("step"+mode+"_task1_"+fname+"_"+minSupport*100+"%_result2.txt", 'w') as f:
        f.write("{0}\n".format(items_len))
        if mode == '2':
            for i, itemNumBefore, itemNumAfter in candidates:
                f.write("{0}\t{1}\t{2}\n".format(i, itemNumBefore, itemNumAfter))

def printResultsFile3(closeItems, minSupport, fname, mode):
    with open("step"+mode+"_task2_"+fname+"_"+minSupport*100+"%_result1.txt", 'w') as f:
        f.write("{0}\n".format(len(closeItems)))
        for item, support in sorted(closeItems, key=lambda x: x[1], reverse=True):
            tmp = str("{0:.1f}%\t{1}\n".format(support*100, item).replace("'", "").replace("(", "{").replace(")", "}"))
            tmp = tmp[::-1].replace(",", "", 1)
            tmp = tmp[::-1]
            f.write(tmp)

# generator
def dataFromFile(fname):
    """Function which reads from the file and yields a generator"""
    with open(fname, "r") as fileIter:
        for line in fileIter:
            line = line.strip().rstrip(',')  # Remove trailing comma
            record = frozenset(line.split(','))
            yield record

# find all closed frequent itemsets
def getClosedItemset(items):
    for item, support in sorted(items, key=lambda x: len(x[0]), reverse=True):
        subsetCombs = set(subsets(item))
        # Remove itself from subsets
        subsetCombs.remove(item)
        for subset in subsetCombs:
            for i, (subsetItem, subsetSup) in enumerate(items):
                # if support of subset equal to itself, then disacrd
                if subset == subsetItem and support == subsetSup:
                    del items[i]
    return items



if __name__ == "__main__":
    # parameters
    optparser = OptionParser()
    optparser.add_option(
        "-m",
        "--mode",
        dest="mode",
        help="2: Apriori algorithm, 3: FP growth",
        default='3',
        type="string",
    )
    (options, args) = optparser.parse_args()
    mode = options.mode
    minSuplist = [[0.05, 0.075, 0.1], 
                  [0.025, 0.05, 0.075], 
                  [0.025, 0.05, 0.075]]
    fileList = [os.path.join("data", "A", "datasetA.csv"), 
                os.path.join("data", "B", "datasetB.csv"), 
                os.path.join("data", "C", "datasetC.csv")]
    fnameList = ['datasetA', 'datasetB', 'datasetC']

    for i in trange(3, desc='processing'):
        for j in range(3):
            start = time.time()
            minSupport = minSuplist[i][j]
            fname = fnameList[i]
            inFile = dataFromFile(fileList[i])


            # -------------- run Apriori --------------
            if mode == '2':

                # -------------- run task1 --------------
                candidatesRatio, items = runApriori(inFile, minSupport)

                end1 = time.time()
                time1 = end1 - start
                printResultsFile1(items, str(minSupport), fname, mode)
                printResultsFile2(candidatesRatio, len(items), str(minSupport), fname, mode)
                print("Task1 exe time([{0}], minSupport[{1}]):\n{2:f} sec".format(fname, minSupport, time1))
            
                # -------------- run task2 --------------
                items = getClosedItemset(items)

                end2 = time.time()
                time2 = end2 - start
                printResultsFile3(items, str(minSupport), fname, mode)
                print("Task2 exe time([{0}], minSupport[{1}]):\n{2:f} sec".format(fname, minSupport, time2))
                print("Ratio(task2 / task1): {0:f}%\n".format((time2 / time1) * 100))


            # -------------- run FPgrowth --------------
            if mode == '3':
                # -------------- run task1 --------------
                clf1 = FPgrowth(min_support=minSupport)
                items = clf1.train(inFile)

                end1 = time.time()
                time1 = end1 - start
                time2 = 0
                printResultsFile1(items, str(minSupport), fname, mode)
                printResultsFile2([], len(items), str(minSupport), fname, mode)
                print("Task1 exe time([{0}], minSupport[{1}]):\n{2:f} sec".format(fname, minSupport, time1))


            # -------------- record execution time --------------
            if i == 0 and j == 0:
                with open('step{0}_time_record.txt'.format(mode), "w") as f:
                    f.write("data: {0}\tminsup: {1}\ttask1 time: {2:f}\ttask2 time: {3:f}\tratio(task2/task1){4:f}\n"\
                        .format(fname, minSupport, time1, time2, (time2 / time1) * 100))
            else:
                with open('step{0}_time_record.txt'.format(mode), "a") as f:
                    f.write("data: {0}\tminsup: {1}\ttask1 time: {2:f}\ttask2 time: {3:f}\tratio(task2/task1){4:f}\n"\
                        .format(fname, minSupport, time1, time2, (time2 / time1) * 100))
    