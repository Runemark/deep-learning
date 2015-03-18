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
    func importArffFile(fileName:String, autoencode:Bool) -> Dataset
    {
        var dataset = Dataset()
        
        var fileURL = NSBundle.mainBundle().URLForResource(fileName, withExtension:"arff")
        if let loadedData = String(contentsOfURL:fileURL!, encoding:NSUTF8StringEncoding, error:nil)
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