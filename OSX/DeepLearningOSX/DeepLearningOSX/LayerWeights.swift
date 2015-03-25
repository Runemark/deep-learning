//
//  Array2D.swift
//  autoencoder
//
//  Created by Martin Mumford on 3/10/15.
//  Copyright (c) 2015 Runemark Studios. All rights reserved.

import Foundation

class LayerWeights
{
    var cols:Int, rows:Int
    var matrix = [[Float]]() // [from][to]
    var reverseMatrix = [[Float]]() // [to][from]
    
    init(fromCount:Int, toCount:Int) {
        
        self.cols = cols
        self.rows = rows
        
        for rowIndex in 0..<rows
        {
            matrix.append([Float()])
            for colIndex in 0..<cols
            {
                matrix[rowIndex].append(Float(0))
            }
        }
        
        for colIndex in 0..<cols
        {
            reverseMatrix.append([Float()])
            for rowIndex in 0..<rows
            {
                matrix[colIndex].append(Float(0))
            }
        }
    }
    
    subscript(from:Int, to:Int) -> Float {
        get
        {
            return matrix[from][to]
        }
        set
        {
            matrix[from][to] = newValue
            reverseMatrix[to][from] = newValue
        }
    }
    
     

}
