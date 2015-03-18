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
        
        println("Hello Data")
        
        var docDirectory = NSSearchPathForDirectoriesInDomains(.DocumentDirectory, .UserDomainMask, true)[0] as String
        
        println("directory: \(docDirectory)")
        
        var importer = DataImporter()
//        var testSet = importer.importArffFile("MNIST_test_500")
        var trainingSet = importer.importArffFile("MNIST_train_500", autoencode:true)

        var autoEncoder = SingleLayerBackpropNet(inputNodes: 784, hiddenNodes:200, outputNodes:784)

        autoEncoder.trainOnDataset(trainingSet, testSet:trainingSet, maxEpochs:1)

        var maximalInputs = autoEncoder.maximalInputsForHiddenNodes()
        
        let windowSize = 28
        
        for (index:Int, maximalInput:[Float]) in enumerate(maximalInputs)
        {
            var maximalWindow = inputVectorToWindow(maximalInput, width:windowSize, height:windowSize)
            UIGraphicsBeginImageContextWithOptions(CGSizeMake(CGFloat(windowSize/2), CGFloat(windowSize/2)), true, 0.0)
            var newImage:UIImage = UIGraphicsGetImageFromCurrentImageContext()
            
            newImage = newImage.getFilledImage(maximalWindow)!

            let targetPath = docDirectory.stringByAppendingPathComponent("derp\(index).png")
            UIImagePNGRepresentation(newImage).writeToFile(targetPath, atomically:false)
            UIGraphicsEndImageContext()
        }
        
        println("everything complete")
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        println("did receive memory warning")
        // Dispose of any resources that can be recreated.
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////
    // VISUALIZATION
    //////////////////////////////////////////////////////////////////////////////////////////
    func inputVectorToWindow(inputVector:[Float], width:Int, height:Int) -> Array2D
    {
        var window = Array2D(cols:width, rows:height)
        
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

