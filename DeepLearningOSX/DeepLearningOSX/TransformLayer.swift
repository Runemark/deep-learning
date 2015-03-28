//
//  SingleLayerBackpropNet.swift
//  Deep-Learning
//
//  Created by Martin Mumford on 3/12/15.
//  Copyright (c) 2015 Runemark Studios. All rights reserved.
//

import Foundation
import Accelerate

class TransformLayer
{
    // Weights
    var firstWeights:Array2D

    var inputCount:Int
    var hiddenCount:Int
    
    var inputActivations:[Float]
    var hiddenActivations:[Float]
    
    init(inputNodes:Int, hiddenNodes:Int, withWeights:Bool, initialFirstWeights:Array2D)
    {
        self.inputCount = inputNodes
        self.hiddenCount = hiddenNodes
        
        self.inputActivations = Array<Float>(count:inputCount+1, repeatedValue:0)
        self.hiddenActivations = Array<Float>(count:hiddenCount+1, repeatedValue:0)
        
        self.firstWeights = Array2D(cols:hiddenCount, rows:inputCount+1)
        
        if (!withWeights)
        {
            initializeWeights()
        }
        else
        {
            initializeWeightsWithWeights(initialFirstWeights)
        }
        
        println("initialization complete")
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////
    // Training
    //////////////////////////////////////////////////////////////////////////////////////////
    
    func calculateActivationsForInstance(featureVector:[Float])
    {
        initializeInputAndBiasActivations(featureVector)
        
        for hiddenIndex in 0..<hiddenCount
        {
            hiddenActivations[hiddenIndex] = calculateActivation(.Hidden, index:hiddenIndex)
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
                let weight = getWeight(inputIndex, toIndex:index)
                net += weight*inputActivations[inputIndex]
            }
            
            return sigmoid(net)
        }
        else
        {
            var net:Float = 0.0
            for hiddenIndex in 0...hiddenCount
            {
                let weight = getWeight(hiddenIndex, toIndex:index)
                net += weight*hiddenActivations[hiddenIndex]
            }
            
            return sigmoid(net)
        }
    }
    
    func sigmoid(value:Float) -> Float
    {
        return Float(Double(1.0) / (Double(1.0) + pow(M_E, -1 * Double(value))))
    }
        
    func getActivation(layer:Layer, index:Int) -> Float
    {
        if (layer == .Input)
        {
            return inputActivations[index]
        }
        else
        {
            return hiddenActivations[index]
        }
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
    }
    
    func initializeWeightsWithWeights(first:Array2D)
    {
        for x in 0..<inputCount+1
        {
            for y in 0..<hiddenCount
            {
                firstWeights[x,y] = first[x,y]
            }
        }
    }
    
    func smallRandomNumber() -> Float
    {
        return ((Float(arc4random()) / Float(UINT32_MAX)) * 0.2) - 0.1
    }
    
    func getWeight(fromIndex:Int, toIndex:Int) -> Float
    {
        return firstWeights[fromIndex,toIndex]
    }
    
    func setWeight(fromIndex:Int, toIndex:Int, value:Float)
    {
        firstWeights[fromIndex,toIndex] = value
    }
}