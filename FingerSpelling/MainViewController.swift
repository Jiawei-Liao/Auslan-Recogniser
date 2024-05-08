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
    
    private var cameraViewClass: CameraView { view as! CameraView }

    @IBOutlet weak var cameraView: UIView!
    @IBOutlet weak var commentsContainerView: UITextView!
    
    let captureSession = AVCaptureSession()
    var previewLayer: AVCaptureVideoPreviewLayer!
    
    weak var databaseController: DatabaseProtocol?
    
    private var handPoseRequest = VNDetectHumanHandPoseRequest()
    
    var currentCameraPosition: AVCaptureDevice.Position = .back
    
    private var circleContainerView: UIView!
    
    let circleViewThumb = CircleView(frame: CGRect(x: 100, y:100, width:200, height: 200))
    let circleViewIndex = CircleView(frame: CGRect(x: 100, y:100, width:200, height: 200))
    let circleViewMiddle = CircleView(frame: CGRect(x: 100, y:100, width:200, height: 200))
    let circleViewRing = CircleView(frame: CGRect(x: 100, y:100, width:200, height: 200))
    let circleViewLittle = CircleView(frame: CGRect(x: 100, y:100, width:200, height: 200))
    let circleViewWrist = CircleView(frame: CGRect(x: 100, y:100, width:200, height: 200))
    
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
        cameraView.layer.addSublayer(previewLayer)
        
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        captureSession.addOutput(videoOutput)
        
        let circleContainerView = UIView(frame: CGRect(x: 0, y: 0, width: 200, height: 200))
        circleContainerView.center = cameraView.center
        cameraView.addSubview(circleContainerView)
        
        handPoseRequest.maximumHandCount = 2
        
        // FOR TESTING ONLY
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(dismissKeyboard))
        view.addGestureRecognizer(tapGesture)
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        //cameraViewClass.previewLayer.session = cameraView
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        captureSession.stopRunning()
        super.viewWillDisappear(animated)
    }

    // MARK: - Capture
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        var thumbTip: CGPoint?
        var indexTip: CGPoint?
        var middleTip: CGPoint?
        var ringTip: CGPoint?
        var littleTip: CGPoint?
        var wrist: CGPoint?
        
        let handler = VNImageRequestHandler(cmSampleBuffer: sampleBuffer, orientation: .up, options: [:])
        do {
            try handler.perform([handPoseRequest])
            
            guard let observations = handPoseRequest.results, observations.count >= 2 else {
                return
            }

            for observation in observations {
                print(try observation.recognizedPoint(VNHumanHandPoseObservation.JointName.wrist))
                
                let thumbPoints = try observation.recognizedPoints(forGroupKey: VNHumanHandPoseObservation.JointsGroupName.thumb.rawValue)
                let indexFingerPoints = try observation.recognizedPoints(forGroupKey: VNHumanHandPoseObservation.JointsGroupName.indexFinger.rawValue)
                let middleFingerPoints = try observation.recognizedPoints(forGroupKey: VNHumanHandPoseObservation.JointsGroupName.middleFinger.rawValue)
                let ringFingerPoints = try observation.recognizedPoints(forGroupKey: VNHumanHandPoseObservation.JointsGroupName.ringFinger.rawValue)
                let littleFingerPoints = try observation.recognizedPoints(forGroupKey: VNHumanHandPoseObservation.JointsGroupName.littleFinger.rawValue)
                let wristPoint = try observation.recognizedPoint(VNHumanHandPoseObservation.JointName.wrist)
                
                guard let thumbTipPoint = thumbPoints[VNRecognizedPointKey(rawValue: "VNHLKTTIP")],
                      let indexTipPoint = indexFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKITIP")],
                      let middleTipPoint = middleFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKMTIP")],
                      let ringTipPoint = ringFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKRTIP")],
                      let littleTipPoint = littleFingerPoints[VNRecognizedPointKey(rawValue: "VNHLKLTIP")]
                else {
                  return
                }
                
                let confidenceThreshold: Float = 0.3
                guard thumbTipPoint.confidence > confidenceThreshold &&
                        indexTipPoint.confidence > confidenceThreshold &&
                        middleTipPoint.confidence > confidenceThreshold &&
                        ringTipPoint.confidence > confidenceThreshold &&
                        littleTipPoint.confidence > confidenceThreshold &&
                        wristPoint.confidence > confidenceThreshold
                else {
                    return
                }
                
                thumbTip = CGPoint(x: thumbTipPoint.location.x, y: 1 - thumbTipPoint.location.y)
                indexTip = CGPoint(x: indexTipPoint.location.x, y: 1 - indexTipPoint.location.y)
                middleTip = CGPoint(x: middleTipPoint.location.x, y: 1 - middleTipPoint.location.y)
                ringTip = CGPoint(x: ringTipPoint.location.x, y: 1 - ringTipPoint.location.y)
                littleTip = CGPoint(x: littleTipPoint.location.x, y: 1 - littleTipPoint.location.y)
                wrist = CGPoint(x: wristPoint.location.x, y: 1 - wristPoint.location.y)
                
                createNewView(circleView: circleViewThumb, fingerTip: thumbTip)
                createNewView(circleView: circleViewIndex, fingerTip: indexTip)
                createNewView(circleView: circleViewMiddle, fingerTip: middleTip)
                createNewView(circleView: circleViewRing, fingerTip: ringTip)
                createNewView(circleView: circleViewLittle, fingerTip: littleTip)
                createNewView(circleView: circleViewWrist, fingerTip: wrist)
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
    
    func createNewView(circleView: CircleView, fingerTip: CGPoint?) {
        if fingerTip != nil {
            circleView.centerPoint = fingerTip!
            circleContainerView.addSubview(circleView)
        }
    }
}
