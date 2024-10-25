//
//  CircleView.swift
//  FingerSpelling
//
//  Created by Shreshth Saharan on 1/5/2024.
//

import Foundation
import UIKit

class CircleView: UIView {
    
    var centerPoint: CGPoint = .zero
    let radius: CGFloat = 10
    
    override func draw(_ rect: CGRect) {
        super.draw(rect)
        
        guard let context = UIGraphicsGetCurrentContext() else { return }
        
        context.setFillColor(UIColor.red.cgColor)
        
        context.fillEllipse(in: CGRect(x: centerPoint.x - radius, y: centerPoint.y - radius, width: radius * 2, height: radius * 2))
    }
}
