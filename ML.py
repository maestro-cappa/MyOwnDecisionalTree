from MLNode import MLNode
from math import pow
import numpy as np
from anytree import Node, RenderTree

class ML:
    def __init__(self, dataset):
        self.dataset = dataset
        self.root = Node("root", my_obj=MLNode(param=-1, paramCol=-1, group=dataset)) #creo un primo nodo con parametro ancora ignoto e l'intero dataset come gruppo
        self.node_counter = 0
        self.score = []

    def train(self, node):

        print(f"\nTRAINING NODE: {node.name}")
        print(f"Group size: {len(node.my_obj.group)}")
        print(f"Gini of current node: {self.gini(node.my_obj.group)}")

        #parto dal primo nodo
        #esamino tutti i possibili gini medi per ogni parametro
        #prendo il parametro con gini medio piu basso e lo salvo
        #divido il gruppo in due sulla base del parametro e lo salvo nel nodo successivo
        #creo nuovo nodo a sinistra e a destra e ricorsivamente richiamo train
        result = self.getLowestGini(node.my_obj.group)
        lowest_gini = result[0] # lowest gini possible with the split of this node
        ideal_param = result[1]
        paramCol = result[2]
        print(f"Lowest Gini found: {lowest_gini} with param {ideal_param} at column {paramCol}")


        current_gini = self.gini(node.my_obj.group)
        if lowest_gini < current_gini: # if new gini isnt better than current gini

            npGroup = np.array(node.my_obj.group)
            mask_left = npGroup[:, paramCol] < ideal_param
            mask_right = ~mask_left
            group_left = npGroup[mask_left].tolist()
            group_right = npGroup[mask_right].tolist()

            if len(group_left) > 0:
                left_child = Node(self.get_unique_name(), my_obj=MLNode(param=-1, paramCol=-1, group=group_left))
                left_child.parent = node
                self.train(left_child)

            if len(group_right) > 0:
                right_child = Node(self.get_unique_name(), my_obj=MLNode(param=-1, paramCol=-1, group=group_right))
                right_child.parent = node
                self.train(right_child)

            node.my_obj.param = ideal_param
            node.my_obj.paramCol = paramCol
            
        else:
            if lowest_gini == 0:
                print("Node is PURE!")
            else:
                print("Node isnt pure but current gini is already the lowest possible!")

            node.my_obj.param = "PURE"
            node.my_obj.paramCol = "PURE"

        
    def guess(self, node, input): # input is an array of the input params, each is associated with a column

        #print(f"\nGUESSING at node: {node.name}")
        #print(f"Node param: {node.my_obj.param}, paramCol: {node.my_obj.paramCol}")

        if node.my_obj.param == "PURE":

            outputClass = int(node.my_obj.group[-1][-1])
            self.score.append(outputClass)

            # print(f"OUTPUT:  {outputClass}") # class is the same for each element in the group, take the last arbitrary
            return
        
        # print(f"{node.name} param: {node.my_obj.param} paramcol: {node.my_obj.paramCol}")

        if input[node.my_obj.paramCol] < node.my_obj.param:
            self.guess(node.children[0], input)
        else:
            self.guess(node.children[1], input)


    def getInput(self):
        inputML = []
        for i in range(self.getNParams(self.root.my_obj.group)):
            inputML.append(float(input(f"Insert {i + 1} param: ")))

        return inputML


    def gini(self, group):
        
        y = 0
        if group:

            classes = self.getClasses(group)
            classesDict = self.getElementsForEachClass(classes)
            if group:   #checking if is empty
                
                for x in classesDict.keys():
                    y += pow(classesDict[x] / len(group), 2)

        return 1 - y
    
    def medium_gini(self, group, param, paramCol):  # media pesata dei gini di ogni sottogruppo
        npGroup = np.array(group)
        mask_left = npGroup[:, paramCol] < param
        mask_right = ~mask_left
        group_left = npGroup[mask_left].tolist()
        group_right = npGroup[mask_right].tolist()

        return len(group_left) / len(group) * self.gini(group_left) + len(group_right) / len(group) * self.gini(group_right)



    def getClasses(self, group):
        npGroup = np.array(group)
        classes = npGroup[:, len(group[0]) - 1]
        return classes.tolist()

    def getElementsForEachClass(self, classes):
        classesDict = {}
        for i in classes:
            if i in classesDict:
                classesDict[i] += 1
            else:
                classesDict[i] = 1
        return classesDict
    
    def getNClasses(self, group):
        return len(set(self.getClasses(group)))
    
    def getParams(self, group):
        npGroup = np.array(group)
        params = np.delete(npGroup, self.getNParams(group), axis=1)
        return params.tolist()

    def getNParams(self, group):
        return len(group[0]) - 1

    def getParamsCol(self, group, paramsCol):
        npGroup = np.array(group)
        params = npGroup[:, paramsCol]

        return params.tolist()
    
    def getLowestGini(self, group):
        step = 0.25
        paramCol = 0
        ideal_param = 0
        lowest_gini = 1

        for i in range(self.getNParams(group)):
            params = self.getParamsCol(group, i)
            param = min(params)
            maxParam = max(params)

            while param < maxParam:
                current_gini = self.medium_gini(group, param, i)
                if current_gini < lowest_gini:
                    lowest_gini = current_gini
                    ideal_param = param
                    paramCol = i
                param += step
        
        
        return [lowest_gini, ideal_param, paramCol]

    def printTree(self):
        for pre, _, node in RenderTree(self.root):
            print(f"{pre}{node.name} - {node.my_obj}")
        
    def isGroupPure(self, group):
        gini = self.gini(group)
        if gini == 0.0:
            return True
        else:
            return False
        
    
    def get_unique_name(self):
        name = f"Node_{self.node_counter}"
        self.node_counter += 1
        return name
            
