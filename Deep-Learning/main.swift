//
//  main.swift
//  Deep Learning
//
//  Created by Martin Mumford on 3/12/15.
//  Copyright (c) 2015 Runemark Studios. All rights reserved.
//

import Foundation

println("Hello Data")

var importer = DataImporter()
var testSet = importer.importArffFile("MNIST_test_500")
var trainingSet = importer.importArffFile("MNIST_train_500")

var net = SingleLayerBackpropNet()
net.trainOnDataset(trainingSet, testSet:testSet, maxEpochs:10)