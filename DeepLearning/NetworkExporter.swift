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

class NetworkExporter
{
    init()
    {
        
    }
    
    // if the half flag is set to true, only the first half of the network is exported (the second half is discarded)
    func exportWeights(network:SingleLayerBackpropNet, half:Bool) -> String
    {
        let inputCount = network.inputCount
        let hiddenCount = network.hiddenCount
        let outputCount = network.outputCount
        
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
            exportString += nodeString
        }
        
        if (!half)
        {
            for hiddenIndex in 0...hiddenCount
            {
                var nodeString = ""
                for outputIndex in 0..<outputCount
                {
                    let weight = secondWeights[hiddenIndex,outputIndex]
                    
                    if (hiddenIndex == outputCount-1)
                    {
                        nodeString += "\(weight),"
                    }
                    else
                    {
                        nodeString += "\(weight)"
                    }
                }
                exportString += nodeString
            }
        }
        
        return exportString
    }
    
//    func transformationLayerFromWeights(weightString:String) -> TransformationLayer
//    {
//        // Takes an input and transforms it into a new space. CANNOT UPDATE.
                // // In order to update, you must add this layer to an updatable network. This LAYER IS UNLOCKABLE.
//    }
    
    func networkFromWeights(weightString:String) -> SingleLayerBackpropNet?
    {
        // Parse out the metadata
      //  var dataset = Dataset()
        
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
            
            if let inputCountValue = Int(metadataComponents[1])
            {
                inputCount = inputCountValue
            }
            
            if let hiddenCountValue = Int(metadataComponents[2])
            {
                hiddenCount = hiddenCountValue
            }
            
            if let outputCountValue = Int(metadataComponents[3])
            {
                outputCount = outputCountValue
            }
            
            if (inputCount > 0 && hiddenCount > 0 && outputCount > 0)
            {
                let net = SingleLayerBackpropNet(inputNodes:inputCount, hiddenNodes:hiddenCount, outputNodes:outputCount)
                
                for line in lines
                {
                    let lineCompareString = line as NSString
                    
                    if lineCompareString.containsString("weights")
                    {
                        if lineCompareString.containsString("second")
                        {
                            weightSet = .second
                        }
                    }
                    else if lineCompareString.containsString("metadata")
                    {
                        // ignore
                    }
                    else
                    {
                        // node string
                        let weightComponents = line.componentsSeparatedByString(",")
                        // Each node string is a list of weights from the source node to each of the destination nodes on a particular layer
                        
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
                            
                            toNodeIndex += 1
                        }
                        
                        fromNodeIndex += 1
                    }
                }
                
                return net
            }
        }
        
        return nil
    }
}