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
    var inputs = [[Float]]()
    var outputs = [[Float]]()
    var instanceCount = 0
    
    init()
    {
        
    }
    
    func addInstance(inputVector:[Float], outputVector:[Float])
    {
        inputs.append(inputVector)
        outputs.append(outputVector)
        instanceCount++
    }
    
    func getInstance(index:Int) -> (features:[Float], targets:[Float])
    {
        return (features:inputs[index], targets:outputs[index])
    }
}