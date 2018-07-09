import numpy as np
import cv2
import json
import os


class Relation:
    def __init__(self):
        self.imagesFileName  = []
        self.numOfFile = 0
        self.imageSize = []
        self.relations = []

    def loadAnnot(self,jsonFile):
        if not os.path.isfile(jsonFile):
            return
        f = open(jsonFile, 'r')
        jsonDict = json.load(f)
        for annot in jsonDict:
            self.imagesFileName.append(annot['name'])
            self.relations.append(annot['relations'])
        self.numOfFile = len(self.imagesFileName)

    def loadRel(self, filename):
        id = self.imagesFileName.index(filename)
        relations = self.relations[id]
        relation =[]
        for rel in relations:
            relation.append((rel['subject'], rel['object'], rel['relation']))
        return relation

#evalute code is coming soon

#get each instance semantic from semanticMap and instanceMap
def getInstanceCate(semanticMap, instanceMap):
    insId = np.unique(instanceMap)
    insId = list(insId[insId != 0])
    insCate = np.zeros(max(insId) + 1).astype(int)
    for id in insId:
        cateList = list(np.unique(semanticMap[instanceMap == id]))
        semanticCateMap = semanticMap.copy()
        semanticCateMap[instanceMap != id] = 0
        if len(cateList) == 1:
            insCate[id] = cateList[0]
        else:
            cateCount = []
            for cate in cateList:
                cateCount.append(np.sum((semanticCateMap == cate).astype(np.int32)))
            insCate[id] = cateList[cateCount.index(max(cateCount))]
    return insCate


#visualize
def visualize_segmentation(segLabelImg, oriImg):
    imgSize = segLabelImg.shape
    colorMap = np.load('segColorMap.npy')
    segId = np.unique(segLabelImg)
    segId = list(segId[segId != 0])
    visualize_seg = np.zeros((imgSize[0], imgSize[1]))
    visualize_seg = visualize_seg.copy()[:, :, np.newaxis]
    visualize_seg = np.tile(visualize_seg, (1, 1, 3))
    for id in segId:
        visualize_seg[segLabelImg == id, ...] = colorMap[id, ...]
    visualize_Map = np.concatenate((oriImg, visualize_seg.astype(np.uint8)), 1)
    cv2.imshow('vis_segmentation', visualize_Map)
    cv2.waitKey(0)

def visualize_instance(insLabelImg, segLabelImg, oriImg):
    semantic_names = [
        "background",
        "human",
        "floor",
        "bed",
        "window",
        "cabinet",
        "door",
        "table",
        "potting-plant",
        "curtain",
        "chair",
        "sofa",
        "shelf",
        "rug",
        "lamp",
        "fridge",
        "stairs",
        "pillow",
        "kitchen-island",
        "sculpture",
        "sink",
        "document",
        "painting/poster",
        "barrel",
        "basket",
        "poke",
        "stool",
        "clothes",
        "bottle",
        "plate",
        "cellphone",
        "toy",
        "cushion",
        "box",
        "display",
        "blanket",
        "pot",
        "nameplate",
        "banners/flag",
        "cup",
        "pen",
        "digital",
        "cooker",
        "umbrella",
        "decoration",
        "straw",
        "certificate",
        "food",
        "club",
        "towel",
        "pet/animals",
        "tool",
        "household-appliances",
        "pram",
        "car/bus/truck",
        "grass",
        "vegetation",
        "water",
        "ground",
        "road",
        "street-light",
        "railing/fence",
        "stand",
        "steps",
        "pillar",
        "awnings/tent",
        "building",
        "mountrain/hill",
        "stone",
        "bridge",
        "bicycle",
        "motorcycle",
        "airplane",
        "boat/ship",
        "balls",
        "swimming-equipment",
        "body-building-apparatus",
        "gun",
        "smoke",
        "rope",
        "amusement-facilities",
        "prop",
        "military-equipment",
        "bag",
        "instruments"
    ]
    semantic = np.unique(segLabelImg)
    semantic = list(semantic[semantic != 0])
    colorMap = np.load('segColorMap.npy')
    grayImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2GRAY)
    grayImg = grayImg.copy()[:, :, np.newaxis]
    grayImg = np.tile(grayImg, (1, 1, 3))
    imgSize = insLabelImg.shape
    for semanticId  in semantic:
        thisCateInsImg = insLabelImg.copy()
        thisCateInsImg[segLabelImg!=semanticId] = 0
        visualize_ins = np.zeros((imgSize[0], imgSize[1]))
        visualize_ins = visualize_ins.copy()[:, :, np.newaxis]
        visualize_ins = np.tile(visualize_ins, (1, 1, 3))
        insId = np.unique(thisCateInsImg)
        insId = list(insId[insId!=0])
        cate_logo = np.ones((100, imgSize[1], 3)).astype(np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        cate_type = semantic_names[semanticId]
        cv2.putText(cate_logo, cate_type, (int(imgSize[1] / 2) - 60, 90), font, 3, (0, 0, 0), 8)
        for i, ins in enumerate(insId):
            visualize_ins[insLabelImg == ins, ...] = colorMap[i+1, ...]
        visualize_temp = grayImg.copy()
        visualize_temp = cv2.addWeighted(visualize_temp.astype(np.uint8), 0.5, visualize_ins.astype(np.uint8), 0.5,0)
        visualize_out = np.concatenate((cate_logo, visualize_temp), 0)
        cv2.imshow('vis_instance', visualize_out)
        cv2.waitKey(0)


def visualize_relation(relation, insLabelImg, oriImg):
    relation_names = [
        'background',
        'hold',
        'touch',
        'drive',
        'eat',
        'drink',
        'play',
        'look',
        'throw',
        'ride',
        'talk',
        'carry',
        'use',
        'pull',
        'push',
        'hit',
        'feed',
        'kick',
        'wear',
        'in-front-of',
        'next-to',
        'on-top-of',
        'behind',
        'on',
        'with',
        'in',
        'sit-on',
        'stand-on',
        'lie-in',
        'squat',
        'other'
    ]
    imgSize = insLabelImg.shape
    subject_color_map = np.array([[0, 0, 255]])
    object_color_map = np.array([[0, 255, 0]])
    grayImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2GRAY)
    grayImg = grayImg.copy()[:, :, np.newaxis]
    grayImg = np.tile(grayImg, (1, 1, 3))

    for rel in relation:
        visualize_rel = np.zeros((imgSize[0], imgSize[1]))
        visualize_rel = visualize_rel.copy()[:, :, np.newaxis]
        visualize_rel = np.tile(visualize_rel, (1, 1, 3))
        visualize_rel[insLabelImg == rel[0], ...] = subject_color_map[0, ...]
        visualize_rel[insLabelImg == rel[1], ...] = object_color_map[0, ...]
        visualize_temp = grayImg.copy()
        visualize_temp = cv2.addWeighted(visualize_temp.astype(np.uint8), 0.5, visualize_rel.astype(np.uint8), 0.5, 0)
        cate_logo = np.ones((100, imgSize[1], 3)).astype(np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        rel_cate = relation_names[rel[2]]
        cv2.putText(cate_logo, rel_cate, (int(imgSize[1]/2) -60,90) , font,3, (0,0,0),8)
        visualize_out = np.concatenate((cate_logo, visualize_temp), 0)
        cv2.imshow('vis_relation', visualize_out)
        cv2.waitKey(0)



