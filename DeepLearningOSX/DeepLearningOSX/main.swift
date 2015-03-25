//
//  main.swift
//  DeepLearningOSX
//
//  Created by Martin Mumford on 3/25/15.
//  Copyright (c) 2015 Martin Mumford. All rights reserved.
//

import Foundation

let importer = DataImporter()
//let trainingSet = importer.importArffFile("MNIST_train_500", autoencode:false)
//let testSet = importer.importArffFile("MNIST_test_500", autoencode:false)
//
//let standardNet = SingleLayerBackpropNet(inputNodes:784, hiddenNodes:200, outputNodes:784)

var trainingSet = Dataset()
trainingSet.addInstance([Float(0.5), Float(0.6)], outputVector:[Float(0.0), Float(1.0)])
trainingSet.addInstance([Float(0.3), Float(0.4)], outputVector:[Float(1.0), Float(0.0)])
trainingSet.addInstance([Float(0.2), Float(0.1)], outputVector:[Float(1.0), Float(0.0)])
trainingSet.addInstance([Float(0.7), Float(0.8)], outputVector:[Float(0.0), Float(1.0)])
trainingSet.addInstance([Float(0.8), Float(0.9)], outputVector:[Float(0.0), Float(1.0)])
trainingSet.addInstance([Float(0.1), Float(0.2)], outputVector:[Float(1.0), Float(0.0)])

var initialFirstWeights = Array2D(cols:2, rows:3)
initialFirstWeights[0,0] = 0.1
initialFirstWeights[0,1] = -0.2
initialFirstWeights[1,0] = 0.3
initialFirstWeights[1,1] = -0.4
initialFirstWeights[2,0] = 0.5
initialFirstWeights[2,1] = -0.6

var initialSecondWeights = Array2D(cols:2, rows:3)
initialFirstWeights[0,0] = 0.1
initialFirstWeights[0,1] = -0.2
initialFirstWeights[1,0] = 0.3
initialFirstWeights[1,1] = -0.4
initialFirstWeights[2,0] = 0.5
initialFirstWeights[2,1] = -0.6

let standardNet = SingleLayerBackpropNet(inputNodes:2, hiddenNodes:2, outputNodes:2, withWeights:true , initialFirstWeights:initialFirstWeights, initialSecondWeights:initialSecondWeights)
let optimizedNet = SingleLayerBackpropNetOptimized(inputNodes:2, hiddenNodes:2, outputNodes:2, withWeights:true , initialFirstWeights:initialFirstWeights, initialSecondWeights:initialSecondWeights)

let startTime = CFAbsoluteTimeGetCurrent()
standardNet.trainOnDataset(trainingSet, testSet:trainingSet, maxEpochs:1000)
let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime

let startTime2 = CFAbsoluteTimeGetCurrent()
optimizedNet.trainOnDataset(trainingSet, testSet:trainingSet, maxEpochs:1000)
let timeElapsed2 = CFAbsoluteTimeGetCurrent() - startTime2

println("Time elapsed for standard network: \(timeElapsed) s")
println("Time elapsed for optimized network: \(timeElapsed2) s")

println("derp")