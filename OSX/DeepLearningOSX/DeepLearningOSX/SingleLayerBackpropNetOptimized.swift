//
//  SingleLayerBackpropNet.swift
//  Deep-Learning
//
//  Created by Martin Mumford on 3/12/15.
//  Copyright (c) 2015 Runemark Studios. All rights reserved.
//

import Foundation
import Accelerate

class SingleLayerBackpropNetOptimized
{
    // Weights
    var firstWeights:Array2D
    var secondWeights:Array2D
    
    var inputActivations:[Float]
    var hiddenActivations:[Float]
    var outputActivations:[Float]
    
    var outputDeltas:[Float]
    var hiddenDeltas:[Float]
    
    var inputCount:Int
    var hiddenCount:Int
    var outputCount:Int
    
    var learningRate:Float = 1.0
    
    init(inputNodes:Int, hiddenNodes:Int, outputNodes:Int, withWeights:Bool, initialFirstWeights:Array2D, initialSecondWeights:Array2D)
    {
        self.inputCount = inputNodes
        self.hiddenCount = hiddenNodes
        self.outputCount = outputNodes
        
        self.inputActivations = Array<Float>(count:inputCount+1, repeatedValue:0)
        self.hiddenActivations = Array<Float>(count:hiddenCount+1, repeatedValue:0)
        self.outputActivations = Array<Float>(count:outputCount, repeatedValue:0)
        
        self.outputDeltas = Array<Float>(count:outputCount, repeatedValue:0)
        self.hiddenDeltas = Array<Float>(count:hiddenCount, repeatedValue:0)
        
        self.firstWeights = Array2D(cols:hiddenCount, rows:inputCount+1)
        self.secondWeights = Array2D(cols:outputCount, rows:hiddenCount+1)
        
        
        if (!withWeights)
        {
            initializeWeights()
        }
        else
        {
            initializeWeightsWithWeights(initialFirstWeights, second:initialFirstWeights)
        }
        
        println("initialization complete")
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////
    // Testing
    //////////////////////////////////////////////////////////////////////////////////////////
    
    func classificationAccuracy(dataset:Dataset) -> Float
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
        
        return Float(correctlyClassifiedInstances)/Float(totalInstances)
    }
    
    // This method is psecific to the MNIST task
    func classificationForInstance(features:[Float]) -> Int
    {
        calculateActivationsForInstance(features)
        
        // Find the output node with the highest activation
        var maxActivation:Float = -1.0
        var indexWithHighestActivation:Int = -1;
        for (outputIndex:Int, activation:Float) in enumerate(outputActivations)
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
    func targetClassification(targetVector:[Float]) -> Int
    {
        var classificationIndex = -1;
        for (index:Int, target:Float) in enumerate(targetVector)
        {
            if (target == 1.0)
            {
                classificationIndex = index
            }
        }
        
        return classificationIndex
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////
    // Maximal Activation
    //////////////////////////////////////////////////////////////////////////////////////////
    
    func maximalInputsForHiddenNodes() -> [[Float]]
    {
        var hiddenNodeInputs = [[Float]]()
        
        for hiddenIndex in 0..<10
            //        for hiddenIndex in 0..<hiddenCount
        {
            println("calculating maximal input for hidden node: \(hiddenIndex)")
            hiddenNodeInputs.append(maximalInputVectorForHiddenNode(hiddenIndex))
            println("done!")
        }
        
        return hiddenNodeInputs
    }
    
    func maximalInputVectorForHiddenNode(nodeIndex:Int) -> [Float]
    {
        var maximalInputVector = Array<Float>(count:inputCount, repeatedValue:0)
        
        for inputIndex in 0..<inputCount
        {
            maximalInputVector[inputIndex] = maximalInputForHiddenNode(nodeIndex, inputIndex:inputIndex)
        }
        
        return maximalInputVector
    }
    
    func maximalInputForHiddenNode(nodeIndex:Int, inputIndex:Int) -> Float
    {
        let weight = getWeight(.Input, fromIndex:inputIndex, toIndex:nodeIndex)
        
        var sumSquaredWeight:Float = 0
        for i in 0..<inputCount
        {
            let inputWeight = getWeight(.Input, fromIndex:i, toIndex:nodeIndex)
            let squaredInputWeight = Float(pow(Double(inputWeight), 2))
            sumSquaredWeight += squaredInputWeight
        }
        
        return Float(weight) / Float(sqrt(Double(sumSquaredWeight)))
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
    
    func trainOnInstance(instance:(features:[Float],targets:[Float]))
    {
        calculateActivationsForInstance(instance.features)
        calculateDeltas(instance.targets)
        applyWeightDeltas()
    }
    
    func calculateActivationsForInstance(featureVector:[Float])
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
    
    func initializeInputAndBiasActivations(featureVector:[Float])
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
    
    func calculateActivation(layer:Layer, index:Int) -> Float
    {
        if (layer == .Hidden)
        {
            var net:Float = 0.0
            for inputIndex in 0...inputCount
            {
                let weight = getWeight(.Input, fromIndex:inputIndex, toIndex:index)
                net += weight*inputActivations[inputIndex]
            }
            
            return sigmoid(net)
        }
        else
        {
            var net:Float = 0.0
            for hiddenIndex in 0...hiddenCount
            {
                let weight = getWeight(.Hidden, fromIndex:hiddenIndex, toIndex:index)
                net += weight*hiddenActivations[hiddenIndex]
            }
            
            return sigmoid(net)
        }
    }
    
    func sigmoid(value:Float) -> Float
    {
        return Float(Double(1.0) / (Double(1.0) + pow(M_E, -1 * Double(value))))
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
    
    func calculateWeightDelta(fromLayer:Layer, fromIndex:Int, toIndex:Int) -> Float
    {
        var nextLayer:Layer = .Output
        if (fromLayer == .Input)
        {
            nextLayer = .Hidden
        }
        
        return learningRate * getActivation(fromLayer, index:fromIndex) * getDelta(nextLayer, index:toIndex)
    }
    
    func calculateDeltas(outputVector:[Float])
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
    
    func calculateOutputDelta(index:Int, target:Float) -> Float
    {
        let actual = getActivation(.Output, index:index)
        return Float(target - actual) * sigmoidDerivative(actual)
    }
    
    func calculateHiddenDelta(index:Int) -> Float
    {
        var weightedSum:Float = 0.0
        for j in 0..<outputCount
        {
            weightedSum += getWeight(.Hidden, fromIndex:index, toIndex:j) * outputDeltas[j]
        }
        
        let activation = getActivation(.Hidden, index:index)
        return weightedSum * sigmoidDerivative(activation)
    }
    
    func getActivation(layer:Layer, index:Int) -> Float
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
    
    func getDelta(layer:Layer, index:Int) -> Float
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
    
    func sigmoidDerivative(value:Float) -> Float
    {
        return value * (1 - value)
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////
    // Weights
    //////////////////////////////////////////////////////////////////////////////////////////
    
    func initializeWeights()
    {
        for x in 0..<inputCount+1
        {
            for y in 0..<hiddenCount
            {
                firstWeights[x,y] = smallRandomNumber()
            }
        }
        
        for x in 0..<hiddenCount+1
        {
            for y in 0..<outputCount
            {
                secondWeights[x,y] = smallRandomNumber()
            }
        }
    }
    
    func initializeWeightsWithWeights(first:Array2D, second:Array2D)
    {
        for x in 0..<inputCount+1
        {
            for y in 0..<hiddenCount
            {
                firstWeights[x,y] = first[x,y]
            }
        }
        
        for x in 0..<hiddenCount+1
        {
            for y in 0..<outputCount
            {
                secondWeights[x,y] = second[x,y]
            }
        }
    }
    
    func smallRandomNumber() -> Float
    {
        return ((Float(arc4random()) / Float(UINT32_MAX)) * 0.2) - 0.1
    }
    
    func getWeight(fromLayer:Layer, fromIndex:Int, toIndex:Int) -> Float
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
    
    func setWeight(fromLayer:Layer, fromIndex:Int, toIndex:Int, value:Float)
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