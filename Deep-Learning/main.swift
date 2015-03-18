//
//  main.swift
//  Deep Learning
//
//  Created by Martin Mumford on 3/12/15.
//  Copyright (c) 2015 Runemark Studios. All rights reserved.
//

import Foundation
import AppKit

println("Hello Data")

var docDirectory = NSSearchPathForDirectoriesInDomains(.DocumentDirectory, .UserDomainMask, true)[0] as String
docDirectory = docDirectory.stringByAppendingPathComponent("SCIENCE_IMAGES")

println("directory: \(docDirectory)")

var importer = DataImporter()
var testSet = importer.importArffFile("MNIST_test_500")
var trainingSet = importer.importArffFile("MNIST_train_500")

var denoisedTrainingSet = importer.denoiseDataset(trainingSet, noiseFrequency:0.2)

var autoEncoder = SingleLayerBackpropNet(inputNodes: 784, hiddenNodes:200, outputNodes:784)

autoEncoder.trainOnDataset(denoisedTrainingSet, testSet:testSet, maxEpochs:1)

var maximalInputs = autoEncoder.maximalInputsForHiddenNodes()

for (index:Int, maximalInput:[Float]) in enumerate(maximalInputs)
{
    
    let rgbColorSpace:CGColorSpaceRef = CGColorSpaceCreateDeviceRGB()
    let context = CGBitmapContextCreate(nil, 28, 28, 8, 0, rgbColorSpace, nil)
    NSBitmapImageRep
    
    let windowDimension = 28
    
    for x in 0..<windowDimension
    {
        for y in 0..<windowDimension
        {
            println("x:\(x), y:\(y)")
            let windowIndex = x*windowDimension + y
            println("windowIndex:\(windowIndex)")
//            let value = maximalInput[windowIndex]
            let value = 0.5
            if (value > 0)
            {
                println("nonzerp:\(value)")
                let color = CGColorCreateGenericRGB(CGFloat(value), CGFloat(value), CGFloat(value), CGFloat(1.0))
                CGContextSetFillColorWithColor(context, color)
                CGContextMoveToPoint(context, CGFloat(x), CGFloat(y))
                CGContextAddRect(context, CGRectMake(CGFloat(x), CGFloat(y), CGFloat(1), CGFloat(1)))
                CGContextFillPath(context)
            }
        }
    }
    
    let path = docDirectory.stringByAppendingPathComponent("node\(index).png")
    var imageProps:NSDictionary = NSDictionary(object:NSNumber(float:1.0), forKey:NSImageCompressionFactor)
    var imageData = imageRep.representationUsingType(NSBitmapImageFileType.NSPNGFileType, properties:imageProps)!
    imageData.writeToFile(path, atomically:true)
}

extension NSImage
{
    func saveTo(path:String)
    {
        var imageData:NSData = self.TIFFRepresentation!
        var imageRep:NSBitmapImageRep = NSBitmapImageRep(data:imageData)!
        
    }
}
