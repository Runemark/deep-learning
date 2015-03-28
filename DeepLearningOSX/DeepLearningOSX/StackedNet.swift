//
//  StackedNet.swift
//  DeepLearningOSX
//
//  Created by Alicia Cicon on 3/27/15.
//  Copyright (c) 2015 Martin Mumford. All rights reserved.
//

import Foundation

class StackedNet {
    
    var transformLayer:TransformLayer
    var trainingLayer:SingleLayerBackpropNet
    
    init(transformLayer:TransformLayer, trainingLayer:SingleLayerBackpropNet)
    {
        self.transformLayer = transformLayer
        self.trainingLayer = trainingLayer
    }
    
    func trainOnDataset(trainSet:Dataset, testSet:Dataset, maxEpochs:Int, maxInstances:Int)
    {
        var instanceLimit = maxInstances
        if (maxInstances < 1 || maxInstances > trainSet.instanceCount)
        {
            instanceLimit = trainSet.instanceCount
        }
        
        for epoch in 0..<maxEpochs
        {
            for index in 0..<instanceLimit
            {
                println("training on instance: \(index)")
                trainOnInstance(trainSet.getInstance(index))
            }
            
            println("epoch \(epoch): \(trainingLayer.classificationAccuracy(testSet))")
        }
    }
    
    func trainOnInstance(instance:(features:[Float],targets:[Float]))
    {
        // Transform the input space
        transformLayer.calculateActivationsForInstance(instance.features)
        
        // Use the output nodes of the transform layer as input nodes to the training layer (create a new, transformed instance)
        let newInstance = (features:transformLayer.hiddenActivations, targets:instance.targets)
        
        trainingLayer.trainOnInstance(newInstance)
    }
    
}