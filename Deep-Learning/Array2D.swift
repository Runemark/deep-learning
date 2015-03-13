//
//  Array2D.swift
//  autoencoder
//
//  Created by Martin Mumford on 3/10/15.
//  Copyright (c) 2015 Runemark Studios. All rights reserved.
//
//  Modified from http://blog.trolieb.com/trouble-multidimensional-arrays-swift/

import Foundation

class Array2D
{
    var cols:Int, rows:Int
    var matrix:[Float]
    
    // (Requires subclassing NSObject)
    //    override var description : String {
    //
    //        var description = "cols = \(cols)\nrows = \(rows)\n"
    //        for x in 0..<rows
    //        {
    //            description += "[\(x)]"
    //            for y in 0..<cols
    //            {
    //                description += " \(y):\(matrix[cols * x + y])"
    //            }
    //            description += "\n"
    //        }
    //
    //        return description
    //    }
    
    init(cols:Int, rows:Int) {
        self.cols = cols
        self.rows = rows
        matrix = Array(count:cols*rows, repeatedValue:0)
    }
    
    subscript(row:Int, col:Int) -> Float {
        get
        {
            return matrix[cols * row + col]
        }
        set
        {
            matrix[cols * row + col] = newValue
        }
    }
    
    func getRow(rowIndex:Int) -> [Float]
    {
        var row = [Float]()
        for colIndex in 0..<cols
        {
            row.append(matrix[cols*rowIndex + colIndex])
        }
        
        return row
    }
    
    func getCol(colIndex:Int) -> [Float]
    {
        var col = [Float]()
        for rowIndex in 0..<rows
        {
            col.append(matrix[cols*rowIndex + colIndex])
        }
        
        return col
    }
    
    func toVector() -> [Float]
    {
        return matrix
    }
    
    func colCount() -> Int {
        return self.cols
    }
    
    func rowCount() -> Int {
        return self.rows
    }
}
