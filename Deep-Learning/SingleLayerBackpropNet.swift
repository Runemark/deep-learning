//
//  SingleLayerBackpropNet.swift
//  Deep-Learning
//
//  Created by Martin Mumford on 3/12/15.
//  Copyright (c) 2015 Runemark Studios. All rights reserved.
//

import Foundation
import Accelerate

enum Layer
{
    case Input, Hidden, Output
}

class SingleLayerBackpropNet
{
    // Weights
//    var firstWeights:Array2D
//    var secondWeights:Array2D
    var firstWeights:Array2D
    var secondWeights:Array2D
    
    var inputActivations:[Double]
    var hiddenActivations:[Double]
    var outputActivations:[Double]
    
    var outputDeltas:[Double]
    var hiddenDeltas:[Double]
    
    var inputCount:Int
    var hiddenCount:Int
    var outputCount:Int
    
    var learningRate:Double = 0.1
    
    init()
    {
        self.inputCount = 784
        self.hiddenCount = 200
        self.outputCount = 10
        
        self.inputActivations = Array<Double>(count:inputCount+1, repeatedValue:0)
        self.hiddenActivations = Array<Double>(count:hiddenCount+1, repeatedValue:0)
        self.outputActivations = Array<Double>(count:outputCount, repeatedValue:0)
        
        self.outputDeltas = Array<Double>(count:outputCount, repeatedValue:0)
        self.hiddenDeltas = Array<Double>(count:hiddenCount, repeatedValue:0)

        self.firstWeights = Array2D(cols:hiddenCount, rows:inputCount+1)
        self.secondWeights = Array2D(cols:outputCount, rows:hiddenCount+1)
        
        println("initialization complete")
    }
    
    // Store weights backwards -- first list the destination nodes in rows, with input nodes in cols
    func initializeWeightSet(rows:Int, cols:Int) -> [[Double]]
    {
        var weightSet = [[Double]]()
        
        for _ in 0..<rows
        {
            var row = [Double]()
            for _ in 0..<cols
            {
                row.append(smallRandomNumber())
            }
            
            weightSet.append(row)
        }
        
        return weightSet
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////
    // Testing
    //////////////////////////////////////////////////////////////////////////////////////////
    
    func classificationAccuracy(dataset:Dataset) -> Double
    {
        let totalInstances = dataset.instanceCount
        var correctlyClassifiedInstances = 0
        // Return classification accuracy
        for index in 0..<totalInstances
        {
            println("testing on instance: \(index)")
            let instance = dataset.getInstance(index)
            let output = classificationForInstance(instance.features)
            let target = targetClassification(instance.targets)
            
            if (output == target)
            {
                correctlyClassifiedInstances++
            }
        }
        
        return Double(correctlyClassifiedInstances)/Double(totalInstances)
    }
    
    // This method is psecific to the MNIST task
    func classificationForInstance(features:[Double]) -> Int
    {
        calculateActivationsForInstance(features)
        
        // Find the output node with the highest activation
        var maxActivation:Double = -1.0
        var indexWithHighestActivation:Int = -1;
        for (outputIndex:Int, activation:Double) in enumerate(outputActivations)
        {
            if activation > maxActivation
            {
                maxActivation = activation
                indexWithHighestActivation = outputIndex
            }
        }
        
        return indexWithHighestActivation
    }
    
    // This method is specific to the MNIST task
    func targetClassification(targetVector:[Double]) -> Int
    {
        var classificationIndex = -1;
        for (index:Int, target:Double) in enumerate(targetVector)
        {
            if (target == 1.0)
            {
                classificationIndex = index
            }
        }
        
        return classificationIndex
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////
    // Training
    //////////////////////////////////////////////////////////////////////////////////////////
    
    func trainOnDataset(trainSet:Dataset, testSet:Dataset, maxEpochs:Int)
    {
        for epoch in 0..<maxEpochs
        {
            for index in 0..<trainSet.instanceCount
            {
                println("training on instance: \(index)")
                trainOnInstance(trainSet.getInstance(index))
            }
            
            println("epoch \(epoch): \(classificationAccuracy(testSet))")
        }
    }
    
    func trainOnInstance(instance:(features:[Double],targets:[Double]))
    {
        calculateActivationsForInstance(instance.features)
        calculateDeltas(instance.targets)
        applyWeightDeltas()
    }
    
    func calculateActivationsForInstance(featureVector:[Double])
    {
        initializeInputAndBiasActivations(featureVector)
        
        for hiddenIndex in 0..<hiddenCount
        {
            hiddenActivations[hiddenIndex] = calculateActivation(.Hidden, index:hiddenIndex)
        }
        
        for outputIndex in 0..<outputCount
        {
            outputActivations[outputIndex] = calculateActivation(.Output, index:outputIndex)
        }
    }
    
    func initializeInputAndBiasActivations(featureVector:[Double])
    {
        // Initialize input activations
        
        for featureIndex in 0..<inputCount
        {
            inputActivations[featureIndex] = featureVector[featureIndex]
        }
        
        // Initialize bias activations
        
        inputActivations[inputCount] = 1
        hiddenActivations[hiddenCount] = 1
    }
    
    func calculateActivation(layer:Layer, index:Int) -> Double
    {
        if (layer == .Hidden)
        {
            var net:Double = 0.0
            for inputIndex in 0...inputCount
            {
                let weight = getWeight(.Input, fromIndex:inputIndex, toIndex:index)
                net += weight*inputActivations[inputIndex]
            }
            
            // Do some time tests here!!!
//            var net = 0.0
//            vDSP_dotprD(inputActivations, 1, firstWeights.getCol(index), 1, &net, vDSP_Length(inputActivations.count))
            
            return sigmoid(net)
        }
        else
        {
            var net:Double = 0.0
            for hiddenIndex in 0...hiddenCount
            {
                let weight = getWeight(.Hidden, fromIndex:hiddenIndex, toIndex:index)
                net += weight*hiddenActivations[hiddenIndex]
            }
            
            // Do some time tests here!!!
//            var net = 0.0
//            vDSP_dotprD(hiddenActivations, 1, secondWeights.getCol(index), 1, &net, vDSP_Length(hiddenActivations.count))
            
            return sigmoid(net)
        }
    }
    
    func sigmoid(value:Double) -> Double
    {
        return Double(1.0 / (1.0 + pow(M_E, -1 * value)))
    }
    
    func applyWeightDeltas()
    {
        // calculate firstWeights delta values (between input and hidden layers)
        for fromWeightIndex in 0...inputCount
        {
            for toWeightIndex in 0..<hiddenCount
            {
                let oldWeightValue = getWeight(.Input, fromIndex:fromWeightIndex, toIndex:toWeightIndex)
                let weightDelta = calculateWeightDelta(.Input, fromIndex:fromWeightIndex, toIndex:toWeightIndex)
                setWeight(.Input, fromIndex:fromWeightIndex, toIndex:toWeightIndex, value:oldWeightValue + weightDelta)
            }
        }
        
        // calculate secondWeights delta values (between hidden and output layers)
        for fromWeightIndex in 0...hiddenCount
        {
            for toWeightIndex in 0..<outputCount
            {
                let oldWeightValue = getWeight(.Hidden, fromIndex:fromWeightIndex, toIndex:toWeightIndex)
                let weightDelta = calculateWeightDelta(.Hidden, fromIndex:fromWeightIndex, toIndex:toWeightIndex)
                setWeight(.Hidden, fromIndex:fromWeightIndex, toIndex:toWeightIndex, value:oldWeightValue + weightDelta)
            }
        }
    }
    
    func calculateWeightDelta(fromLayer:Layer, fromIndex:Int, toIndex:Int) -> Double
    {
        var nextLayer:Layer = .Output
        if (fromLayer == .Input)
        {
            nextLayer = .Hidden
        }
        
        return learningRate * getActivation(fromLayer, index:fromIndex) * getDelta(nextLayer, index:toIndex)
    }
    
    func calculateDeltas(outputVector:[Double])
    {
        for outputIndex in 0..<outputCount
        {
            outputDeltas[outputIndex] = calculateOutputDelta(outputIndex, target:outputVector[outputIndex])
        }
        
        for hiddenIndex in 0..<hiddenCount
        {
            hiddenDeltas[hiddenIndex] = calculateHiddenDelta(hiddenIndex)
        }
    }
    
    func calculateOutputDelta(index:Int, target:Double) -> Double
    {
        let actual = getActivation(.Output, index:index)
        return (target - actual) * sigmoidDerivative(actual)
    }
    
    func calculateHiddenDelta(index:Int) -> Double
    {
        var weightedSum:Double = 0.0
        for j in 0..<outputCount
        {
            weightedSum += getWeight(.Hidden, fromIndex:index, toIndex:j) * outputDeltas[j]
        }
        
        let activation = getActivation(.Hidden, index:index)
        return weightedSum * sigmoidDerivative(activation)
    }
    
    func getActivation(layer:Layer, index:Int) -> Double
    {
        if (layer == .Input)
        {
            return inputActivations[index]
        }
        else if (layer == .Hidden)
        {
            return hiddenActivations[index]
        }
        else
        {
            return outputActivations[index]
        }
    }
    
    func getDelta(layer:Layer, index:Int) -> Double
    {
        if (layer == .Output)
        {
            return outputDeltas[index]
        }
        else
        {
            return hiddenDeltas[index]
        }
    }
    
    func sigmoidDerivative(value:Double) -> Double
    {
        return value * (1 - value)
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////
    // Weights
    //////////////////////////////////////////////////////////////////////////////////////////
    
    func smallRandomNumber() -> Double
    {
        return ((Double(arc4random()) / Double(UINT32_MAX)) * 0.2) - 0.1
    }
    
    func getWeight(fromLayer:Layer, fromIndex:Int, toIndex:Int) -> Double
    {
        if (fromLayer == .Input)
        {
            return firstWeights[fromIndex,toIndex]
        }
        else
        {
            return secondWeights[fromIndex,toIndex]
        }
    }
    
    func setWeight(fromLayer:Layer, fromIndex:Int, toIndex:Int, value:Double)
    {
        if (fromLayer == .Input)
        {
            firstWeights[fromIndex,toIndex] = value
        }
        else
        {
            secondWeights[fromIndex,toIndex] = value
        }
    }
}