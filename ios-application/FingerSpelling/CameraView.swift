//
//  CameraView.swift
//  FingerSpelling
//
//  Adapted from https://github.com/r4ghu/iOS-Vision-HandPose
//

import UIKit
import AVFoundation

class CameraView: UIView {
    
    private var overlayThumbLayer = CAShapeLayer()
    private var overlayIndexLayer = CAShapeLayer()
    private var overlayMiddleLayer = CAShapeLayer()
    private var overlayRingLayer = CAShapeLayer()
    private var overlayLittleLayer = CAShapeLayer()
    
    var previewLayer: AVCaptureVideoPreviewLayer!
    
    override class var layerClass: AnyClass {
        return AVCaptureVideoPreviewLayer.self
    }
    
    override init(frame: CGRect) {
        super.init(frame: frame)
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
    }
    
    override func layoutSublayers(of layer: CALayer) {
        super.layoutSublayers(of: layer)
        if layer == previewLayer {
            overlayThumbLayer.frame = layer.bounds
            overlayIndexLayer.frame = layer.bounds
            overlayMiddleLayer.frame = layer.bounds
            overlayRingLayer.frame = layer.bounds
            overlayLittleLayer.frame = layer.bounds
        }
    }
    
    func setupOverlay() {
        previewLayer.addSublayer(overlayThumbLayer)
        previewLayer.addSublayer(overlayIndexLayer)
        previewLayer.addSublayer(overlayMiddleLayer)
        previewLayer.addSublayer(overlayRingLayer)
        previewLayer.addSublayer(overlayLittleLayer)
    }
    
    func showPoints(_ points: [CGPoint]) {
        clearLayers()
        
        guard points.count >= 21 else {
            return
        }

        let wrist: CGPoint = points.last!
        let thumbColor = UIColor.green
        let indexColor = UIColor.blue
        let middleColor = UIColor.yellow
        let ringColor = UIColor.cyan
        let littleColor = UIColor.red
        
        drawFinger(overlayThumbLayer, Array(points[0...4]), thumbColor, wrist)
        drawFinger(overlayIndexLayer, Array(points[4...8]), indexColor, wrist)
        drawFinger(overlayMiddleLayer, Array(points[8...12]), middleColor, wrist)
        drawFinger(overlayRingLayer, Array(points[12...16]), ringColor, wrist)
        drawFinger(overlayLittleLayer, Array(points[16...20]), littleColor, wrist)
    }
    
    func drawFinger(_ layer: CAShapeLayer, _ points: [CGPoint], _ color: UIColor, _ wrist: CGPoint) {
        let fingerPath = UIBezierPath()
        
        for point in points {
            fingerPath.move(to: point)
            fingerPath.addArc(withCenter: point, radius: 5, startAngle: 0, endAngle: 2 * .pi, clockwise: true)
        }
        
        fingerPath.move(to: points[0])
        fingerPath.addLine(to: points[1])
        fingerPath.move(to: points[1])
        fingerPath.addLine(to: points[2])
        fingerPath.move(to: points[2])
        fingerPath.addLine(to: points[3])
        fingerPath.move(to: points[3])
        fingerPath.addLine(to: wrist)
        
        layer.fillColor = color.cgColor
        layer.strokeColor = color.cgColor
        layer.lineWidth = 5.0
        layer.lineCap = .round
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        layer.path = fingerPath.cgPath
        CATransaction.commit()
    }
    
    func clearLayers() {
        let emptyPath = UIBezierPath()
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        overlayThumbLayer.path = emptyPath.cgPath
        overlayIndexLayer.path = emptyPath.cgPath
        overlayMiddleLayer.path = emptyPath.cgPath
        overlayRingLayer.path = emptyPath.cgPath
        overlayLittleLayer.path = emptyPath.cgPath
        CATransaction.commit()
    }
}
