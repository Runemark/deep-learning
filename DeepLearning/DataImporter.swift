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

        var loadedData:String
        do{
            try loadedData =  String(contentsOfFile:path, encoding:NSUTF8StringEncoding)
        }
        catch let error as NSError {
            print(error.localizedDescription)
            
            return nil
        }
        return loadedData
        
    }

}

class DataImporter
{
    func importArffFile(fileName:String, autoencode:Bool) -> Dataset
    {
        let dataset = Dataset()
        
        let fileURL = NSBundle.mainBundle().URLForResource(fileName, withExtension:"arff")
        var loadedData:String = String()
        do{
            try loadedData =  String(contentsOfURL:fileURL!, encoding:NSUTF8StringEncoding)
        }
        catch let error as NSError {
            print(error.localizedDescription)

        }

        
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
                        let value:Int? = Int(element)
                        
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
                    
                    instanceCount += 1
                    
                    if (instanceCount % 10 == 0)
                    {
                        print("loading instance: \(instanceCount)")
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
                        featureCount += 1
                    }
                    else if (line as NSString).containsString("class")
                    {
                        let classComponents = line.componentsSeparatedByString("{")
                        let classesString = classComponents[1].stringByReplacingOccurrencesOfString("}", withString:"", options:[], range:nil)
                        let classes = classesString.componentsSeparatedByString(",")
                        classCount = classes.count
                    }
                }
            }
        
        
        return dataset
    }
    
    func denoiseDataset(dataset:Dataset, noiseFrequency:Double) -> Dataset
    {
        let denoisedDataset = Dataset()
        
        for instanceIndex in 0..<dataset.instanceCount
        {
            let instance = dataset.getInstance(instanceIndex)
            let originalInput = instance.features
            
            var modifiedInput = [Float]()
            var modifiedOutput = [Float]()
            
            for featureIndex in 0..<originalInput.count
            {
                let feature = originalInput[featureIndex]
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