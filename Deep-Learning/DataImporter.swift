//
//  DataImporter.swift
//  Deep Learning
//
//  Created by Martin Mumford on 3/12/15.
//  Copyright (c) 2015 Runemark Studios. All rights reserved.
//

import Foundation

class File {
    class func open (path: String, utf8: NSStringEncoding = NSUTF8StringEncoding) -> String? {
        var error: NSError?
        return NSFileManager().fileExistsAtPath(path) ? String(contentsOfFile: path, encoding: utf8, error: &error)! : nil
    }
}

class DataImporter
{
    func importArffFile(fileName:String) -> Dataset
    {
        var directory = "\(NSHomeDirectory())/Documents/Research/deep-learning/Data/\(fileName).arff"
        
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
                    var instance = (features:[Double](), targets:[Double]())
                    
                    for (index:Int, element:String) in enumerate(components)
                    {
                        // Should be a number from 0 to 255
                        if let value = element.toInt()
                        {
                            if (index < featureCount)
                            {
                                instance.features.append(Double(value)/255.0)
                            }
                            else
                            {
                                for n in 0..<classCount
                                {
                                    if n == value
                                    {
                                        instance.targets.append(Double(1))
                                    }
                                    else
                                    {
                                        instance.targets.append(Double(0))
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
                    
                    dataset.addInstance(instance.features, outputVector:instance.targets)
                }
                else
                {
                    if (line as NSString).containsString("@DATA")
                    {
                        dataSection = true
                    }
                    else if (line as NSString).containsString("real")
                    {
                        featureCount++
                    }
                    else if (line as NSString).containsString("class")
                    {
                        let classComponents = line.componentsSeparatedByString("{")
                        let classesString = classComponents[1].stringByReplacingOccurrencesOfString("}", withString:"", options:nil, range:nil)
                        let classes = classesString.componentsSeparatedByString(",")
                        classCount = classes.count
                    }
                }
            }
            
            println("features: \(featureCount)")
            println("Data Import Complete")
        }
        else
        {
            println("file not loaded: \(fileName)")
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
            
            var modifiedInput = [Double]()
            var modifiedOutput = [Double]()
            
            for featureIndex in 0..<originalInput.count
            {
                var feature = originalInput[featureIndex]
                modifiedOutput.append(feature)
                
                if (randomWithProbability(noiseFrequency))
                {
                    modifiedInput.append(feature*randNormalizedDouble())
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
    
    func randNormalizedDouble() -> Double {
        let randInt = randRange(0, upper:1000)
        return Double(randInt)/1000.0
    }
    
    func randomWithProbability(probability:Double) -> Bool
    {
        let randInt = randRange(0, upper:100)
        return Double(randInt)/100.0 < probability
    }
}