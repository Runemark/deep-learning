//
//  NetworkExporter.swift
//  Deep-Learning
//
//  Created by Martin Mumford on 3/13/15.
//  Copyright (c) 2015 Runemark Studios. All rights reserved.
//

import Foundation

class NetworkExporter
{
    init()
    {
        
    }
    
    func exportWeights(network:SingleLayerBackpropNet) -> String
    {
        var inputCount = network.inputCount
        var hiddenCount = network.hiddenCount
        var outputCount = network.outputCount
        
        let firstWeights = network.firstWeights
        let secondWeights = network.secondWeights
        
        var exportString = "\(inputCount):\(hiddenCount):\(outputCount)\n"
        exportString += "weights:first\n"
        
        for inputIndex in 0...inputCount // includes bias weight
        {
            var nodeString = "inputnode:\(inputIndex)\n"
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
        
        for hiddenIndex in 0...hiddenCount
        {
            var nodeString = "hiddennode:\(hiddenIndex)"
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
        
        return exportString
    }
    
//    func networkWithWeights(weightString:String) -> SingleLayerBackpropNet
//    {
//        
//    }
}