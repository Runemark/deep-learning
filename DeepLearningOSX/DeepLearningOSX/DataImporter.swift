//
//  DataImporter.swift
//  Deep Learning
//
//  Created by Martin Mumford on 3/12/15.
//  Copyright (c) 2015 Runemark Studios. All rights reserved.
//

import Foundation

class DataImporter
{
    func importArffFile(fileName:String, autoencode:Bool) -> Dataset
    {
        var directory = "\(NSHomeDirectory())/Desktop/deep-learning/Data/\(fileName).arff"
        
        var dataset = Dataset()
        
        if let loadedData = String(contentsOfFile:directory, encoding:NSUTF8StringEncoding, error:nil)
        {
            let lines = loadedData.componentsSeparatedByString("\n")
            
            var dataSection = false
            var featureCount = 0
            var instanceCount = 0
            var classCount = 0
            
            for line in lines
            {
                if (dataSection && !line.isEmpty)
                {
                    let components = line.componentsSeparatedByString(",")
                    var instanceFeatures = [Float]()
                    var instanceTargets = [Float]()
                    
                    for index in 0..<components.count
                    {
                        let element = components[index]
                        let value:Int? = element.toInt()
                        
                        if value != nil
                        {
                            if (index < featureCount)
                            {
                                let normalizedValue = Float(value!)/Float(255.0)
                                instanceFeatures.append(normalizedValue)
                                
                                if (autoencode)
                                {
                                    instanceTargets.append(normalizedValue)
                                }
                            }
                            else if (!autoencode)
                            {
                                for n in 0..<classCount
                                {
                                    if n == value
                                    {
                                        instanceTargets.append(Float(1.0))
                                    }
                                    else
                                    {
                                        instanceTargets.append(Float(0.0))
                                    }
                                }
                            }
                        }
                    }
                    
                    instanceCount++
                    
                    if (instanceCount % 10 == 0)
                    {
                        println("loading instance: \(instanceCount)")
                    }
                    
                    dataset.addInstance(instanceFeatures, outputVector:instanceTargets)
                }
                else
                {
                    if line.rangeOfString("@DATA") != nil
                    {
                        dataSection = true
                    }
                    else if line.rangeOfString("real") != nil
                    {
                        featureCount++
                    }
                    else if line.rangeOfString("class") != nil
                    {
                        let classComponents = line.componentsSeparatedByString("{")
                        let classesString = classComponents[1].stringByReplacingOccurrencesOfString("}", withString:"", options:nil, range:nil)
                        let classes = classesString.componentsSeparatedByString(",")
                        classCount = classes.count
                    }
                }
            }
        }
        
        return dataset
    }
    
    func denoiseDataset(dataset:Dataset, noiseFrequency:Double) -> Dataset
    {
        var denoisedDataset = Dataset()
        
        for instanceIndex in 0..<dataset.instanceCount
        {
            let instance = dataset.getInstance(instanceIndex)
            let originalInput = instance.features
            
            var modifiedInput = [Float]()
            var modifiedOutput = [Float]()
            
            for featureIndex in 0..<originalInput.count
            {
                var feature = originalInput[featureIndex]
                modifiedOutput.append(feature)
                
                if (randomWithProbability(noiseFrequency))
                {
                    modifiedInput.append(feature*randNormalizedFloat())
                }
                else
                {
                    modifiedInput.append(feature)
                }
            }
            
            denoisedDataset.addInstance(modifiedInput, outputVector:modifiedOutput)
        }
        
        return denoisedDataset
    }
    
    func randRange (lower:Int , upper:Int) -> Int {
        return lower + Int(arc4random_uniform(UInt32(upper - lower + 1)))
    }
    
    func randNormalizedFloat() -> Float {
        let randInt = randRange(0, upper:1000)
        return Float(randInt)/Float(1000.0)
    }
    
    func randomWithProbability(probability:Double) -> Bool
    {
        let randInt = randRange(0, upper:100)
        return Float(randInt)/100.0 < Float(probability)
    }
}