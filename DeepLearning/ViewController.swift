//
//  ViewController.swift
//  DeepLearning
//
//  Created by Martin Mumford on 3/17/15.
//  Copyright (c) 2015 Martin Mumford. All rights reserved.
//

import UIKit

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        print("Hello Data")
        
        let docDirectory = NSSearchPathForDirectoriesInDomains(.DocumentDirectory, .UserDomainMask, true)[0] as String
        
        print("directory: \(docDirectory)")
        
        let importer = DataImporter()
        
        //////////////////////////////////////////////////////////////////////////////////////////
        // AUTOENCODER
        //////////////////////////////////////////////////////////////////////////////////////////
        
//        var testSet = importer.importArffFile("MNIST_test_500")
//        var trainingSet = importer.importArffFile("MNIST_train_500", autoencode:true)
//        
//        var autoEncoder = SingleLayerBackpropNet(inputNodes: 784, hiddenNodes:500, outputNodes:784)
//        autoEncoder.trainOnDataset(trainingSet, testSet:trainingSet, maxEpochs:1)
//        
//        var maximalInputs = autoEncoder.maximalInputsForHiddenNodes()
        
        //////////////////////////////////////////////////////////////////////////////////////////
        
        
        //////////////////////////////////////////////////////////////////////////////////////////
        // STANDARD BACKPROP (784:200:10)
        //////////////////////////////////////////////////////////////////////////////////////////
        let testSet = importer.importArffFile("MNIST_test_500", autoencode:false)
        let trainingSet = importer.importArffFile("MNIST_train_500", autoencode:false)
        
        let standardBackprop = SingleLayerBackpropNet(inputNodes:784, hiddenNodes:200, outputNodes:10)
        standardBackprop.trainOnDataset(trainingSet, testSet:testSet, maxEpochs:10)
        //////////////////////////////////////////////////////////////////////////////////////////
        
        
        //////////////////////////////////////////////////////////////////////////////////////////
        // TEST BACKPROP (2:2:2)
        //////////////////////////////////////////////////////////////////////////////////////////
        
//        var testSet = Dataset()
//        testSet.addInstance([Float(0.5), Float(0.6)], outputVector:[Float(1.0), Float(0.0)])
//        testSet.addInstance([Float(0.2), Float(0.3)], outputVector:[Float(0.0), Float(1.0)])
//        
//        var standardBackprop = SingleLayerBackpropNet(inputNodes:2, hiddenNodes:2, outputNodes:2)
//        standardBackprop.trainOnDataset(testSet, testSet:testSet, maxEpochs:1)
        
        //////////////////////////////////////////////////////////////////////////////////////////
        
//        let windowSize = 28
//        
//        for (index:Int, maximalInput:[Float]) in enumerate(maximalInputs)
//        {
//            var maximalWindow = inputVectorToWindow(maximalInput, width:windowSize, height:windowSize)
//            UIGraphicsBeginImageContextWithOptions(CGSizeMake(CGFloat(windowSize/2), CGFloat(windowSize/2)), true, 0.0)
//            var newImage:UIImage = UIGraphicsGetImageFromCurrentImageContext()
//            
//            newImage = newImage.getFilledImage(maximalWindow)!
//
//            let targetPath = docDirectory.stringByAppendingPathComponent("derp\(index).png")
//            UIImagePNGRepresentation(newImage).writeToFile(targetPath, atomically:false)
//            UIGraphicsEndImageContext()
//        }
        
        print("everything complete")
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        print("did receive memory warning")
        // Dispose of any resources that can be recreated.
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////
    // VISUALIZATION
    //////////////////////////////////////////////////////////////////////////////////////////
    func inputVectorToWindow(inputVector:[Float], width:Int, height:Int) -> Array2D
    {
        let window = Array2D(cols:width, rows:height)
        
        for x in 0..<height
        {
            for y in 0..<width
            {
                window[x,y] = inputVector[x*width+y]
            }
        }
        
        return window
    }
    
//    func writeWindowToImage(window:Array2D, inout image:UIImage)
//    {
//        for x in 0..<window.rowCount()
//        {
//            for y in 0..<window.colCount()
//            {
//                let value8Bit = Int(floor(Double(window[x,y]*255)))
//                image.setPixelAlphaAtPoint(CGPointMake(CGFloat(x),CGFloat(y)), alpha:UInt8(value8Bit))
//            }
//        }
//    }
}

