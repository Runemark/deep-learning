//
//  NetworkExporter.swift
//  Deep-Learning
//
//  Created by Martin Mumford on 3/13/15.
//  Copyright (c) 2015 Runemark Studios. All rights reserved.
//

import Foundation

enum WeightSet {
    case first, second
}

extension String {
    var floatValue: Float {
        return (self as NSString).floatValue
    }
}

class NetworkIO
{
    init()
    {
        
    }
    
    // if the half flag is set to true, only the first half of the network is exported (the second half is discarded)
    func exportWeights(network:SingleLayerBackpropNet, half:Bool) -> String
    {
        var inputCount = network.inputCount
        var hiddenCount = network.hiddenCount
        var outputCount = network.outputCount
        
        let firstWeights = network.firstWeights
        let secondWeights = network.secondWeights
        
        var exportString = "metadata:\(inputCount):\(hiddenCount):\(outputCount)\n"
        exportString += "weights:first\n"
        
        for inputIndex in 0...inputCount // includes bias weight
        {
            var nodeString = ""
            for hiddenIndex in 0..<hiddenCount
            {
                let weight = firstWeights[inputIndex,hiddenIndex]
                
                if (hiddenIndex == hiddenCount-1)
                {
                    nodeString += "\(weight)"
                }
                else
                {
                    nodeString += "\(weight),"
                }
            }
            
            exportString += nodeString + "\n"
            
        }
        
        if (!half)
        {
            exportString += "weights:second\n"
            for hiddenIndex in 0...hiddenCount
            {
                var nodeString = ""
                for outputIndex in 0..<outputCount
                {
                    let weight = secondWeights[hiddenIndex,outputIndex]
                    
                    if (outputIndex == outputCount-1)
                    {
                        nodeString += "\(weight)"
                    }
                    else
                    {
                        nodeString += "\(weight),"
                    }
                }
                
                if (hiddenIndex == hiddenCount)
                {
                    exportString += nodeString
                }
                else
                {
                    exportString += nodeString + "\n"
                }
            }
        }
        
        return exportString
    }
    
    func networkFromFile(fileName:NSString) -> SingleLayerBackpropNet?
    {
        var directory = "\(NSHomeDirectory())/Documents/Academics/CS678-NeuralNetworks/Project2-DeepLearning/SavedNets/\(fileName).txt"
        
        var dataset = Dataset()
        
        if let loadedData = String(contentsOfFile:directory, encoding:NSUTF8StringEncoding, error:nil)
        {
            return networkFromWeights(loadedData)
        }
        
        return nil
    }
    
    func halfNetworkFromFile(fileName:NSString) -> TransformLayer?
    {
        var directory = "\(NSHomeDirectory())/Documents/Academics/CS678-NeuralNetworks/Project2-DeepLearning/SavedNets/\(fileName).txt"
        
        var dataset = Dataset()
        
        if let loadedData = String(contentsOfFile:directory, encoding:NSUTF8StringEncoding, error:nil)
        {
            return halfNetworkFromWeights(loadedData)
        }
        
        return nil
    }
    
    func halfNetworkFromWeights(weightString:String) -> TransformLayer?
    {
        let lines = weightString.componentsSeparatedByString("\n")
        if (lines.count > 0)
        {
            var inputCount = 0
            var hiddenCount = 0
            var outputCount = 0
            
            var weightSet:WeightSet = .first
            var fromNodeIndex = 0
            var toNodeIndex = 0
            
            let metadata = lines[0]
            let metadataComponents = metadata.componentsSeparatedByString(":")
            
            if let inputCountValue = metadataComponents[1].toInt()
            {
                inputCount = inputCountValue
            }
            
            if let hiddenCountValue = metadataComponents[2].toInt()
            {
                hiddenCount = hiddenCountValue
            }
            
            if let outputCountValue = metadataComponents[3].toInt()
            {
                outputCount = outputCountValue
            }
            
            if (inputCount > 0 && hiddenCount > 0 && outputCount > 0)
            {
                let net = TransformLayer(inputNodes:inputCount, hiddenNodes:hiddenCount, withWeights:false, initialFirstWeights:Array2D(cols:1, rows:1))
                
                for line in lines
                {
                    if line.rangeOfString("weights") != nil
                    {
                        if line.rangeOfString("second") != nil
                        {
                            weightSet = .second
                            fromNodeIndex = 0
                            toNodeIndex = 0
                        }
                    }
                    else if line.rangeOfString("metadata") != nil
                    {
                        // ignore
                    }
                    else
                    {
                        // node string
                        let weightComponents = line.componentsSeparatedByString(",")
                        // Each node string is a list of weights from the source node to each of the destination nodes on a particular layer
                        
                        toNodeIndex = 0
                        
                        for weightComponent in weightComponents
                        {
                            let weightValue:Float = weightComponent.floatValue
                            
                            if (weightSet == .first)
                            {
                                net.firstWeights[fromNodeIndex,toNodeIndex] = weightValue
                            }
                            
                            toNodeIndex++
                        }
                        
                        fromNodeIndex++
                    }
                }
                
                return net
            }
        }
        
        return nil
    }
    
    func networkFromWeights(weightString:String) -> SingleLayerBackpropNet?
    {
        let lines = weightString.componentsSeparatedByString("\n")
        if (lines.count > 0)
        {
            var inputCount = 0
            var hiddenCount = 0
            var outputCount = 0
            
            var weightSet:WeightSet = .first
            var fromNodeIndex = 0
            var toNodeIndex = 0
            
            let metadata = lines[0]
            let metadataComponents = metadata.componentsSeparatedByString(":")
            
            if let inputCountValue = metadataComponents[1].toInt()
            {
                inputCount = inputCountValue
            }
            
            if let hiddenCountValue = metadataComponents[2].toInt()
            {
                hiddenCount = hiddenCountValue
            }
            
            if let outputCountValue = metadataComponents[3].toInt()
            {
                outputCount = outputCountValue
            }
            
            if (inputCount > 0 && hiddenCount > 0 && outputCount > 0)
            {
                let net = SingleLayerBackpropNet(inputNodes:inputCount, hiddenNodes:hiddenCount, outputNodes:outputCount, withWeights:false, initialFirstWeights:Array2D(cols:1, rows:1), initialSecondWeights:Array2D(cols:1, rows:1))
                
                for line in lines
                {
                    if line.rangeOfString("weights") != nil
                    {
                        if line.rangeOfString("second") != nil
                        {
                            weightSet = .second
                            fromNodeIndex = 0
                            toNodeIndex = 0
                        }
                    }
                    else if line.rangeOfString("metadata") != nil
                    {
                        // ignore
                    }
                    else
                    {
                        // node string
                        let weightComponents = line.componentsSeparatedByString(",")
                        // Each node string is a list of weights from the source node to each of the destination nodes on a particular layer
                        
                        toNodeIndex = 0
                        
                        for weightComponent in weightComponents
                        {
                            let weightValue:Float = weightComponent.floatValue
                            
                            if (weightSet == .first)
                            {
                                net.firstWeights[fromNodeIndex,toNodeIndex] = weightValue
                            }
                            else
                            {
                                net.secondWeights[fromNodeIndex,toNodeIndex] = weightValue
                            }
                            
                            toNodeIndex++
                        }
                        
                        fromNodeIndex++
                    }
                }
                
                return net
            }
        }
        
        return nil
    }
}