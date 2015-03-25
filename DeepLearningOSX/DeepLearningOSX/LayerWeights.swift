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
    var transpose = [[Float]]() // [to][from]
    
    init(fromCount:Int, toCount:Int) {
        
        self.cols = toCount
        self.rows = fromCount
        
        for rowIndex in 0..<rows
        {
            matrix.append([Float()])
            for colIndex in 0..<cols-1
            {
                matrix[rowIndex].append(Float(0))
            }
        }
        
        for colIndex in 0..<cols
        {
            transpose.append([Float()])
            for rowIndex in 0..<rows-1
            {
                transpose[colIndex].append(Float(0))
            }
        }
    }
    
    func weightsFrom(index:Int) -> [Float]
    {
        return matrix[index]
    }
    
    func weightsTo(index:Int) -> [Float]
    {
        return transpose[index]
    }
    
    subscript(from:Int, to:Int) -> Float {
        get
        {
            return matrix[from][to]
        }
        set
        {
            matrix[from][to] = newValue
            transpose[to][from] = newValue
        }
    }
    
     

}
