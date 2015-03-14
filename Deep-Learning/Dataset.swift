//
//  Dataset.swift
//  Deep-Learning
//
//  Created by Martin Mumford on 3/12/15.
//  Copyright (c) 2015 Runemark Studios. All rights reserved.
//

import Foundation

class Dataset
{
    var inputs = [[Double]]()
    var outputs = [[Double]]()
    var instanceCount = 0
    
    init()
    {
        
    }
    
    func addInstance(inputVector:[Double], outputVector:[Double])
    {
        inputs.append(inputVector)
        outputs.append(outputVector)
        instanceCount++
    }
    
    func getInstance(index:Int) -> (features:[Double], targets:[Double])
    {
        return (features:inputs[index], targets:outputs[index])
    }
}