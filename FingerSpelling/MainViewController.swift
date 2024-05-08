//
//  ViewController.swift
//  FingerSpelling
//
//  Created by Shreshth Saharan on 27/3/2024.
//

import UIKit
import AVFoundation
import Vision

class MainViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    // MARK: - HAND ONE
    // Thumb
    var thumbTipHandOne: CGPoint?
    var thumbIpHandOne: CGPoint?
    var thumbMpHandOne: CGPoint?
    var thumbCmcHandOne: CGPoint?
    
    // Index
    var indexTipHandOne: CGPoint?
    var indexDipHandOne: CGPoint?
    var indexPipHandOne: CGPoint?
    var indexMcpHandOne: CGPoint?
    
    // Middle
    var middleTipHandOne: CGPoint?
    var middleDipHandOne: CGPoint?
    var middlePipHandOne: CGPoint?
    var middleMcpHandOne: CGPoint?
    
    // Ring
    var ringTipHandOne: CGPoint?
    var ringDipHandOne: CGPoint?
    var ringPipHandOne: CGPoint?
    var ringMcpHandOne: CGPoint?
    
    //Little
    var littleTipHandOne: CGPoint?
    var littleDipHandOne: CGPoint?
    var littlePipHandOne: CGPoint?
    var littleMcpHandOne: CGPoint?
    
    // Wrist
    var wristHandOne: CGPoint?
    
    // MARK: - HAND TWO
    // Thumb
    var thumbTipHandTwo: CGPoint?
    var thumbIpHandTwo: CGPoint?
    var thumbMpHandTwo: CGPoint?
    var thumbCmcHandTwo: CGPoint?

    // Index
    var indexTipHandTwo: CGPoint?
    var indexDipHandTwo: CGPoint?
    var indexPipHandTwo: CGPoint?
    var indexMcpHandTwo: CGPoint?

    // Middle
    var middleTipHandTwo: CGPoint?
    var middleDipHandTwo: CGPoint?
    var middlePipHandTwo: CGPoint?
    var middleMcpHandTwo: CGPoint?

    // Ring
    var ringTipHandTwo: CGPoint?
    var ringDipHandTwo: CGPoint?
    var ringPipHandTwo: CGPoint?
    var ringMcpHandTwo: CGPoint?

    //Little
    var littleTipHandTwo: CGPoint?
    var littleDipHandTwo: CGPoint?
    var littlePipHandTwo: CGPoint?
    var littleMcpHandTwo: CGPoint?

    // Wrist
    var wristHandTwo: CGPoint?
    
    // MARK: - Other variables / IBActions
    
    private var cameraViewClass: CameraView { self.cameraView as! CameraView }
    
    @IBOutlet weak var cameraView: UIView!
    @IBOutlet weak var commentsContainerView: UITextView!
    
    let captureSession = AVCaptureSession()
    var previewLayer: AVCaptureVideoPreviewLayer!
    
    private var handPoseRequest = VNDetectHumanHandPoseRequest()
    var currentCameraPosition: AVCaptureDevice.Position = .back
    
    weak var databaseController: DatabaseProtocol?
    
    @IBAction func clearEntry(_ sender: Any) {
        commentsContainerView.text = ""
    }
    
    @IBAction func saveEntry(_ sender: Any) {
        guard let text = commentsContainerView.text else {
            return
        }
        
        if text.isEmpty {
            let errorMsg = "Please ensure there is some available text"
            
            displayMessage(title: "Empty translation", message: errorMsg)
            return
        }
        
        let _ = databaseController?.addEntry(content: text)
    }
    
    // MARK: - viewDidLoad method
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let appDelegate = UIApplication.shared.delegate as? AppDelegate
        databaseController = appDelegate?.databaseController
        
        let settings = databaseController?.fetchSettings()
        
        if ((settings?.frontCamera) != nil) {
            currentCameraPosition = .front
        }
        
        guard let captureDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: currentCameraPosition) else { return }
        
        guard let input = try? AVCaptureDeviceInput(device: captureDevice) else { return }
        
        captureSession.addInput(input)
        captureSession.startRunning()
        
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = cameraView.bounds
        
        // Passing current preview layer to the grapher (which is in CameraView.swift)
        cameraViewClass.previewLayer = previewLayer
        cameraViewClass.setupOverlay()
        
        cameraView.layer.addSublayer(previewLayer)
        
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        captureSession.addOutput(videoOutput)
        
        handPoseRequest.maximumHandCount = 2
        
        // FOR TESTING ONLY
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(dismissKeyboard))
        view.addGestureRecognizer(tapGesture)
    }
    
    // MARK: - Capture
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        let handler = VNImageRequestHandler(cmSampleBuffer: sampleBuffer, orientation: .up, options: [:])
        do {
            try handler.perform([handPoseRequest])
            
            guard let observations = handPoseRequest.results, observations.count == 2 else {
                cameraViewClass.showPoints([])
                return
            }
            
            var iterationOne = true
            
            for observation in observations {
                let thumbPoints = try observation.recognizedPoints(forGroupKey: VNHumanHandPoseObservation.JointsGroupName.thumb.rawValue)
                let indexFingerPoints = try observation.recognizedPoints(forGroupKey: VNHumanHandPoseObservation.JointsGroupName.indexFinger.rawValue)
                let middleFingerPoints = try observation.recognizedPoints(forGroupKey: VNHumanHandPoseObservation.JointsGroupName.middleFinger.rawValue)
                let ringFingerPoints = try observation.recognizedPoints(forGroupKey: VNHumanHandPoseObservation.JointsGroupName.ringFinger.rawValue)
                let littleFingerPoints = try observation.recognizedPoints(forGroupKey: VNHumanHandPoseObservation.JointsGroupName.littleFinger.rawValue)
                let wristPoint = try observation.recognizedPoint(VNHumanHandPoseObservation.JointName.wrist)
                
                if iterationOne {
                    guard let thumbTipPoint = thumbPoints[VNRecognizedPointKey(rawValue: "VNHLKTTIP")],
                          let thumbIpPoint = thumbPoints[VNRecognizedPointKey(rawValue: "VNHLKTIP")],
                          let thumbMpPoint = thumbPoints[VNRecognizedPointKey(rawValue: "VNHLKTMP")],
                          let thumbCmcPoint = thumbPoints[VNRecognizedPointKey(rawValue: "VNHLKTCMC")],
                          let indexTipPoint = indexFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKITIP")],
                          let indexDipPoint = indexFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKIDIP")],
                          let indexPipPoint = indexFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKIPIP")],
                          let indexMcpPoint = indexFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKIMCP")],
                          let middleTipPoint = middleFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKMTIP")],
                          let middleDipPoint = middleFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKMDIP")],
                          let middlePipPoint = middleFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKMPIP")],
                          let middleMcpPoint = middleFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKMMCP")],
                          let ringTipPoint = ringFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKRTIP")],
                          let ringDipPoint = ringFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKRDIP")],
                          let ringPipPoint = ringFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKRPIP")],
                          let ringMcpPoint = ringFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKRMCP")],
                          let littleTipPoint = littleFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKPTIP")],
                          let littleDipPoint = littleFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKPDIP")],
                          let littlePipPoint = littleFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKPPIP")],
                          let littleMcpPoint = littleFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKPMCP")]
                    else {
                        cameraViewClass.showPoints([])
                        print("Failed point extraction")
                        return
                    }
                    
                    let confidenceThreshold: Float = 0.3
                    guard thumbTipPoint.confidence > confidenceThreshold &&
                            thumbIpPoint.confidence > confidenceThreshold &&
                            thumbMpPoint.confidence > confidenceThreshold &&
                            thumbCmcPoint.confidence > confidenceThreshold &&
                            indexTipPoint.confidence > confidenceThreshold &&
                            indexDipPoint.confidence > confidenceThreshold &&
                            indexPipPoint.confidence > confidenceThreshold &&
                            indexMcpPoint.confidence > confidenceThreshold &&
                            middleTipPoint.confidence > confidenceThreshold &&
                            middleDipPoint.confidence > confidenceThreshold &&
                            middlePipPoint.confidence > confidenceThreshold &&
                            middleMcpPoint.confidence > confidenceThreshold &&
                            ringTipPoint.confidence > confidenceThreshold &&
                            ringDipPoint.confidence > confidenceThreshold &&
                            ringPipPoint.confidence > confidenceThreshold &&
                            ringMcpPoint.confidence > confidenceThreshold &&
                            littleTipPoint.confidence > confidenceThreshold &&
                            littleDipPoint.confidence > confidenceThreshold &&
                            littlePipPoint.confidence > confidenceThreshold &&
                            littleMcpPoint.confidence > confidenceThreshold &&
                            wristPoint.confidence > confidenceThreshold
                    else {
                        cameraViewClass.showPoints([])
                        print("Failed confidence test")
                        return
                    }
                    
                    thumbTipHandOne = CGPoint(x: thumbTipPoint.location.x, y: 1 - thumbTipPoint.location.y)
                    thumbIpHandOne = CGPoint(x: thumbIpPoint.location.x, y: 1 - thumbIpPoint.location.y)
                    thumbMpHandOne = CGPoint(x: thumbMpPoint.location.x, y: 1 - thumbMpPoint.location.y)
                    thumbCmcHandOne = CGPoint(x: thumbCmcPoint.location.x, y: 1 - thumbCmcPoint.location.y)
                    indexTipHandOne = CGPoint(x: indexTipPoint.location.x, y: 1 - indexTipPoint.location.y)
                    indexDipHandOne = CGPoint(x: indexDipPoint.location.x, y: 1 - indexDipPoint.location.y)
                    indexPipHandOne = CGPoint(x: indexPipPoint.location.x, y: 1 - indexPipPoint.location.y)
                    indexMcpHandOne = CGPoint(x: indexMcpPoint.location.x, y: 1 - indexMcpPoint.location.y)
                    middleTipHandOne = CGPoint(x: middleTipPoint.location.x, y: 1 - middleTipPoint.location.y)
                    middleDipHandOne = CGPoint(x: middleDipPoint.location.x, y: 1 - middleDipPoint.location.y)
                    middlePipHandOne = CGPoint(x: middlePipPoint.location.x, y: 1 - middlePipPoint.location.y)
                    middleMcpHandOne = CGPoint(x: middleMcpPoint.location.x, y: 1 - middleMcpPoint.location.y)
                    ringTipHandOne = CGPoint(x: ringTipPoint.location.x, y: 1 - ringTipPoint.location.y)
                    ringDipHandOne = CGPoint(x: ringDipPoint.location.x, y: 1 - ringDipPoint.location.y)
                    ringPipHandOne = CGPoint(x: ringPipPoint.location.x, y: 1 - ringPipPoint.location.y)
                    ringMcpHandOne = CGPoint(x: ringMcpPoint.location.x, y: 1 - ringMcpPoint.location.y)
                    littleTipHandOne = CGPoint(x: littleTipPoint.location.x, y: 1 - littleTipPoint.location.y)
                    littleDipHandOne = CGPoint(x: littleDipPoint.location.x, y: 1 - littleDipPoint.location.y)
                    littlePipHandOne = CGPoint(x: littlePipPoint.location.x, y: 1 - littlePipPoint.location.y)
                    littleMcpHandOne = CGPoint(x: littleMcpPoint.location.x, y: 1 - littleMcpPoint.location.y)
                    wristHandOne = CGPoint(x: wristPoint.location.x, y: 1 - wristPoint.location.y)
                    
                    DispatchQueue.main.async {
                        self.convertPoints([self.thumbTipHandOne, self.thumbIpHandOne, self.thumbMpHandOne, self.thumbCmcHandOne, self.indexTipHandOne, self.indexDipHandOne, self.indexPipHandOne, self.indexMcpHandOne, self.middleTipHandOne, self.middleDipHandOne, self.middlePipHandOne, self.middleMcpHandOne, self.ringTipHandOne, self.ringDipHandOne, self.ringPipHandOne, self.ringMcpHandOne, self.littleTipHandOne, self.littleDipHandOne, self.littlePipHandOne, self.littleMcpHandOne, self.wristHandOne])
                    }

                    iterationOne = false
                    
                } else {
                    guard let thumbTipPoint = thumbPoints[VNRecognizedPointKey(rawValue: "VNHLKTTIP")],
                          let thumbIpPoint = thumbPoints[VNRecognizedPointKey(rawValue: "VNHLKTIP")],
                          let thumbMpPoint = thumbPoints[VNRecognizedPointKey(rawValue: "VNHLKTMP")],
                          let thumbCmcPoint = thumbPoints[VNRecognizedPointKey(rawValue: "VNHLKTCMC")],
                          let indexTipPoint = indexFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKITIP")],
                          let indexDipPoint = indexFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKIDIP")],
                          let indexPipPoint = indexFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKIPIP")],
                          let indexMcpPoint = indexFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKIMCP")],
                          let middleTipPoint = middleFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKMTIP")],
                          let middleDipPoint = middleFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKMDIP")],
                          let middlePipPoint = middleFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKMPIP")],
                          let middleMcpPoint = middleFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKMMCP")],
                          let ringTipPoint = ringFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKRTIP")],
                          let ringDipPoint = ringFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKRDIP")],
                          let ringPipPoint = ringFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKRPIP")],
                          let ringMcpPoint = ringFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKRMCP")],
                          let littleTipPoint = littleFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKPTIP")],
                          let littleDipPoint = littleFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKPDIP")],
                          let littlePipPoint = littleFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKPPIP")],
                          let littleMcpPoint = littleFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKPMCP")]
                    else {
                        cameraViewClass.showPoints([])
                        print("Failed point extraction")
                        return
                    }
                    
                    let confidenceThreshold: Float = 0.3
                    guard thumbTipPoint.confidence > confidenceThreshold &&
                            thumbIpPoint.confidence > confidenceThreshold &&
                            thumbMpPoint.confidence > confidenceThreshold &&
                            thumbCmcPoint.confidence > confidenceThreshold &&
                            indexTipPoint.confidence > confidenceThreshold &&
                            indexDipPoint.confidence > confidenceThreshold &&
                            indexPipPoint.confidence > confidenceThreshold &&
                            indexMcpPoint.confidence > confidenceThreshold &&
                            middleTipPoint.confidence > confidenceThreshold &&
                            middleDipPoint.confidence > confidenceThreshold &&
                            middlePipPoint.confidence > confidenceThreshold &&
                            middleMcpPoint.confidence > confidenceThreshold &&
                            ringTipPoint.confidence > confidenceThreshold &&
                            ringDipPoint.confidence > confidenceThreshold &&
                            ringPipPoint.confidence > confidenceThreshold &&
                            ringMcpPoint.confidence > confidenceThreshold &&
                            littleTipPoint.confidence > confidenceThreshold &&
                            littleDipPoint.confidence > confidenceThreshold &&
                            littlePipPoint.confidence > confidenceThreshold &&
                            littleMcpPoint.confidence > confidenceThreshold &&
                            wristPoint.confidence > confidenceThreshold
                    else {
                        cameraViewClass.showPoints([])
                        print("Failed confidence test")
                        return
                    }
                    
                    thumbTipHandTwo = CGPoint(x: thumbTipPoint.location.x, y: 1 - thumbTipPoint.location.y)
                    thumbIpHandTwo = CGPoint(x: thumbIpPoint.location.x, y: 1 - thumbIpPoint.location.y)
                    thumbMpHandTwo = CGPoint(x: thumbMpPoint.location.x, y: 1 - thumbMpPoint.location.y)
                    thumbCmcHandTwo = CGPoint(x: thumbCmcPoint.location.x, y: 1 - thumbCmcPoint.location.y)
                    indexTipHandTwo = CGPoint(x: indexTipPoint.location.x, y: 1 - indexTipPoint.location.y)
                    indexDipHandTwo = CGPoint(x: indexDipPoint.location.x, y: 1 - indexDipPoint.location.y)
                    indexPipHandTwo = CGPoint(x: indexPipPoint.location.x, y: 1 - indexPipPoint.location.y)
                    indexMcpHandTwo = CGPoint(x: indexMcpPoint.location.x, y: 1 - indexMcpPoint.location.y)
                    middleTipHandTwo = CGPoint(x: middleTipPoint.location.x, y: 1 - middleTipPoint.location.y)
                    middleDipHandTwo = CGPoint(x: middleDipPoint.location.x, y: 1 - middleDipPoint.location.y)
                    middlePipHandTwo = CGPoint(x: middlePipPoint.location.x, y: 1 - middlePipPoint.location.y)
                    middleMcpHandTwo = CGPoint(x: middleMcpPoint.location.x, y: 1 - middleMcpPoint.location.y)
                    ringTipHandTwo = CGPoint(x: ringTipPoint.location.x, y: 1 - ringTipPoint.location.y)
                    ringDipHandTwo = CGPoint(x: ringDipPoint.location.x, y: 1 - ringDipPoint.location.y)
                    ringPipHandTwo = CGPoint(x: ringPipPoint.location.x, y: 1 - ringPipPoint.location.y)
                    ringMcpHandTwo = CGPoint(x: ringMcpPoint.location.x, y: 1 - ringMcpPoint.location.y)
                    littleTipHandTwo = CGPoint(x: littleTipPoint.location.x, y: 1 - littleTipPoint.location.y)
                    littleDipHandTwo = CGPoint(x: littleDipPoint.location.x, y: 1 - littleDipPoint.location.y)
                    littlePipHandTwo = CGPoint(x: littlePipPoint.location.x, y: 1 - littlePipPoint.location.y)
                    littleMcpHandTwo = CGPoint(x: littleMcpPoint.location.x, y: 1 - littleMcpPoint.location.y)
                    wristHandTwo = CGPoint(x: wristPoint.location.x, y: 1 - wristPoint.location.y)

                    DispatchQueue.main.async {
                        self.convertPoints([self.thumbTipHandTwo, self.thumbIpHandTwo, self.thumbMpHandTwo, self.thumbCmcHandTwo, self.indexTipHandTwo, self.indexDipHandTwo, self.indexPipHandTwo, self.indexMcpHandTwo, self.middleTipHandTwo, self.middleDipHandTwo, self.middlePipHandTwo, self.middleMcpHandTwo, self.ringTipHandTwo, self.ringDipHandTwo, self.ringPipHandTwo, self.ringMcpHandTwo, self.littleTipHandTwo, self.littleDipHandTwo, self.littlePipHandTwo, self.littleMcpHandTwo, self.wristHandTwo])
                    }
                }
            }
            
        } catch {
            captureSession.stopRunning()
            DispatchQueue.main.async {
                let alertController = UIAlertController(title: "Error", message: (error as? LocalizedError)?.errorDescription ?? "An unknown error occurred.", preferredStyle: .alert)
                alertController.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))
                self.present(alertController, animated: true, completion: nil)
            }
        }
    }
    
    // MARK: - Other functions
    func displayMessage(title: String, message: String) {
        let alertController = UIAlertController(title: title, message: message, preferredStyle: .alert)
        alertController.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))
        present(alertController, animated: true, completion: nil)
    }
    
    @objc func dismissKeyboard() {
        view.endEditing(true)
    }
    
    func convertPoints(_ points: [CGPoint?]) {
        var pointsDiscovered: [CGPoint] = []
        
        for point in points {
            pointsDiscovered.append(cameraViewClass.previewLayer.layerPointConverted(fromCaptureDevicePoint: point!))
        }
        
        cameraViewClass.showPoints(pointsDiscovered)
        print("Showing points now....")
    }
}
