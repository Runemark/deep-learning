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
        var directory = "\(NSHomeDirectory())/Documents/Research/deep-learning/Deep-Learning/Data/\(fileName).arff"
        
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
                    var instance = (features:[Float](), targets:[Float]())
                    
                    for (index:Int, element:String) in enumerate(components)
                    {
                        // Should be a number from 0 to 255
                        if let value = element.toInt()
                        {
                            if (index < featureCount)
                            {
                                instance.features.append(Float(value)/255.0)
                            }
                            else
                            {
                                for n in 0..<classCount
                                {
                                    if n == value
                                    {
                                        instance.targets.append(Float(1))
                                    }
                                    else
                                    {
                                        instance.targets.append(Float(0))
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
}