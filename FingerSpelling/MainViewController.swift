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
    
    private var cameraViewClass: CameraView { self.cameraView as! CameraView }
    
    @IBOutlet weak var cameraView: UIView!
    @IBOutlet weak var commentsContainerView: UITextView!
    
    let captureSession = AVCaptureSession()
    var previewLayer: AVCaptureVideoPreviewLayer!
    
    private var handPoseRequest = VNDetectHumanHandPoseRequest()
    var currentCameraPosition: AVCaptureDevice.Position = .back
    
    weak var databaseController: DatabaseProtocol?
    
//    private var circleContainerView: UIView!
//    let circleViewThumb = CircleView(frame: CGRect(x: 100, y:100, width:200, height: 200))
//    let circleViewIndex = CircleView(frame: CGRect(x: 100, y:100, width:200, height: 200))
//    let circleViewMiddle = CircleView(frame: CGRect(x: 100, y:100, width:200, height: 200))
//    let circleViewRing = CircleView(frame: CGRect(x: 100, y:100, width:200, height: 200))
//    let circleViewLittle = CircleView(frame: CGRect(x: 100, y:100, width:200, height: 200))
//    let circleViewWrist = CircleView(frame: CGRect(x: 100, y:100, width:200, height: 200))
    
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
        // TESTING
        cameraViewClass.previewLayer = previewLayer
        cameraViewClass.setupOverlay()
        
        cameraView.layer.addSublayer(previewLayer)
        
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        captureSession.addOutput(videoOutput)
        
        // For drawing the joints
        let circleContainerView = UIView(frame: CGRect(x: 0, y: 0, width: 200, height: 200))
        circleContainerView.center = cameraView.center
        cameraView.addSubview(circleContainerView)
        
        handPoseRequest.maximumHandCount = 2
        
        // FOR TESTING ONLY
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(dismissKeyboard))
        view.addGestureRecognizer(tapGesture)
    }
    
    func convertPoints(_ points: [CGPoint?]) {
        var pointsDiscovered: [CGPoint] = []
        
        for point in points {
            pointsDiscovered.append(cameraViewClass.previewLayer.layerPointConverted(fromCaptureDevicePoint: point!))
        }
        
        cameraViewClass.showPoints(pointsDiscovered)
        print("Showing points now....")
    }
    
    // MARK: - Capture
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        // Thumb
        var thumbTip: CGPoint?
        var thumbIp: CGPoint?
        var thumbMp: CGPoint?
        var thumbCmc: CGPoint?
        
        // Index
        var indexTip: CGPoint?
        var indexDip: CGPoint?
        var indexPip: CGPoint?
        var indexMcp: CGPoint?
        
        // Middle
        var middleTip: CGPoint?
        var middleDip: CGPoint?
        var middlePip: CGPoint?
        var middleMcp: CGPoint?
        
        // Ring
        var ringTip: CGPoint?
        var ringDip: CGPoint?
        var ringPip: CGPoint?
        var ringMcp: CGPoint?
        
        //Little
        var littleTip: CGPoint?
        var littleDip: CGPoint?
        var littlePip: CGPoint?
        var littleMcp: CGPoint?
        
        // Wrist
        var wrist: CGPoint?
        
        let handler = VNImageRequestHandler(cmSampleBuffer: sampleBuffer, orientation: .up, options: [:])
        do {
            try handler.perform([handPoseRequest])
            
            guard let observations = handPoseRequest.results, observations.count >= 2 else {
                cameraViewClass.showPoints([])
                return
            }
            
            for observation in observations {
                print(observation.availableJointNames)
                let thumbPoints = try observation.recognizedPoints(forGroupKey: VNHumanHandPoseObservation.JointsGroupName.thumb.rawValue)
                let indexFingerPoints = try observation.recognizedPoints(forGroupKey: VNHumanHandPoseObservation.JointsGroupName.indexFinger.rawValue)
                let middleFingerPoints = try observation.recognizedPoints(forGroupKey: VNHumanHandPoseObservation.JointsGroupName.middleFinger.rawValue)
                let ringFingerPoints = try observation.recognizedPoints(forGroupKey: VNHumanHandPoseObservation.JointsGroupName.ringFinger.rawValue)
                let littleFingerPoints = try observation.recognizedPoints(forGroupKey: VNHumanHandPoseObservation.JointsGroupName.littleFinger.rawValue)
                let wristPoint = try observation.recognizedPoint(VNHumanHandPoseObservation.JointName.wrist)
                
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
                
                thumbTip = CGPoint(x: thumbTipPoint.location.x, y: 1 - thumbTipPoint.location.y)
                thumbIp = CGPoint(x: thumbIpPoint.location.x, y: 1 - thumbIpPoint.location.y)
                thumbMp = CGPoint(x: thumbMpPoint.location.x, y: 1 - thumbMpPoint.location.y)
                thumbCmc = CGPoint(x: thumbCmcPoint.location.x, y: 1 - thumbCmcPoint.location.y)
                indexTip = CGPoint(x: indexTipPoint.location.x, y: 1 - indexTipPoint.location.y)
                indexDip = CGPoint(x: indexDipPoint.location.x, y: 1 - indexDipPoint.location.y)
                indexPip = CGPoint(x: indexPipPoint.location.x, y: 1 - indexPipPoint.location.y)
                indexMcp = CGPoint(x: indexMcpPoint.location.x, y: 1 - indexMcpPoint.location.y)
                middleTip = CGPoint(x: middleTipPoint.location.x, y: 1 - middleTipPoint.location.y)
                middleDip = CGPoint(x: middleDipPoint.location.x, y: 1 - middleDipPoint.location.y)
                middlePip = CGPoint(x: middlePipPoint.location.x, y: 1 - middlePipPoint.location.y)
                middleMcp = CGPoint(x: middleMcpPoint.location.x, y: 1 - middleMcpPoint.location.y)
                ringTip = CGPoint(x: ringTipPoint.location.x, y: 1 - ringTipPoint.location.y)
                ringDip = CGPoint(x: ringDipPoint.location.x, y: 1 - ringDipPoint.location.y)
                ringPip = CGPoint(x: ringPipPoint.location.x, y: 1 - ringPipPoint.location.y)
                ringMcp = CGPoint(x: ringMcpPoint.location.x, y: 1 - ringMcpPoint.location.y)
                littleTip = CGPoint(x: littleTipPoint.location.x, y: 1 - littleTipPoint.location.y)
                littleDip = CGPoint(x: littleDipPoint.location.x, y: 1 - littleDipPoint.location.y)
                littlePip = CGPoint(x: littlePipPoint.location.x, y: 1 - littlePipPoint.location.y)
                littleMcp = CGPoint(x: littleMcpPoint.location.x, y: 1 - littleMcpPoint.location.y)
                wrist = CGPoint(x: wristPoint.location.x, y: 1 - wristPoint.location.y)
                
                DispatchQueue.main.async {
                    self.convertPoints([thumbTip, thumbIp, thumbMp, thumbCmc,indexTip, indexDip, indexPip, indexMcp, middleTip, middleDip, middlePip, middleMcp, ringTip, ringDip, ringPip, ringMcp, littleTip, littleDip, littlePip, littleMcp, wrist])
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
}
