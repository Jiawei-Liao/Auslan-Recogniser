//
//  ViewController.swift
//  FingerSpelling
//
//  Created by Shreshth Saharan on 27/3/2024.
//

import UIKit
import AVFoundation
import Vision
import CoreML
import TensorFlowLite

class MainViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    // MARK: - Other variables / IBActions
    
    private var cameraViewClass: CameraView { self.cameraView as! CameraView }
    
    @IBOutlet weak var cameraView: UIView!
    @IBOutlet weak var commentsContainerView: UITextView!
    
    let captureSession = AVCaptureSession()
    var previewLayer: AVCaptureVideoPreviewLayer!
    
    private var handPoseRequest = VNDetectHumanHandPoseRequest()
    var currentCameraPosition: AVCaptureDevice.Position = .back
    
    weak var databaseController: DatabaseProtocol?
    
    var totalFrames: [[Float32]] = []
    
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
    
    // Helper function to extract points
    func extractPoints(from observation: VNHumanHandPoseObservation, forHand hand: inout [VNRecognizedPointKey: CGPoint], confidenceThreshold: Float) -> Bool {
        do {
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
                    //cameraViewClass.showPoints([])
                    print("Failed point extraction")
                    return false
                }
            
            let pointKeys = [
                thumbTipPoint, thumbIpPoint, thumbMpPoint, thumbCmcPoint,
                indexTipPoint, indexDipPoint, indexPipPoint, indexMcpPoint,
                middleTipPoint, middleDipPoint, middlePipPoint, middleMcpPoint,
                ringTipPoint, ringDipPoint, ringPipPoint, ringMcpPoint,
                littleTipPoint, littleDipPoint, littlePipPoint, littleMcpPoint
            ]

            for point in pointKeys {
                if point.confidence > confidenceThreshold {
                    hand[point.identifier] = CGPoint(x: point.location.x, y: 1 - point.location.y)
                } else {
                    return false
                }
            }

            if wristPoint.confidence > confidenceThreshold {
                hand[VNRecognizedPointKey(rawValue: "wrist")] = CGPoint(x: wristPoint.location.x, y: 1 - wristPoint.location.y)
            } else {
                return false
            }

            return true
            
        } catch {
            print("Error extracting points: \(error)")
            return false
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let handler = VNImageRequestHandler(cmSampleBuffer: sampleBuffer, orientation: .up, options: [:])
        do {
            try handler.perform([handPoseRequest])
            
            guard let observations = handPoseRequest.results, observations.count > 1 else {
                DispatchQueue.main.async {
                    //self.cameraViewClass.showPoints([])
                }
                return
            }

            var handOne: [VNRecognizedPointKey: CGPoint] = [:]
            var handTwo: [VNRecognizedPointKey: CGPoint] = [:]

            let confidenceThreshold: Float = 0.3

            let handOneSuccess = extractPoints(from: observations[0], forHand: &handOne, confidenceThreshold: confidenceThreshold)
            
            var handTwoSuccess = false

            if observations.count > 1 {
                handTwoSuccess = extractPoints(from: observations[1], forHand: &handTwo, confidenceThreshold: confidenceThreshold)
            }

            if !handOneSuccess || (observations.count > 1 && !handTwoSuccess) {
                DispatchQueue.main.async {
                    self.cameraViewClass.showPoints([])
                }
                print("Failed point extraction or confidence test")
                return
            }

            if observations.count == 2 {
                if let handOneWrist = handOne[VNRecognizedPointKey(rawValue: "wrist")],
                   let handTwoWrist = handTwo[VNRecognizedPointKey(rawValue: "wrist")] {

                    let handOneIsLeftHand = handOneWrist.x < handTwoWrist.x

                    let pointsLeftHand = handOneIsLeftHand ? handOne.values : handTwo.values
                    let pointsRightHand = handOneIsLeftHand ? handTwo.values : handOne.values

                    DispatchQueue.main.async {
                        let pointsLeftHandConverted = pointsLeftHand.map { self.cameraViewClass.previewLayer.layerPointConverted(fromCaptureDevicePoint: $0)
                        }
                        
                        let pointsRightHandConverted = pointsRightHand.map { self.cameraViewClass.previewLayer.layerPointConverted(fromCaptureDevicePoint: $0)
                        }
                        
                        print("Left hand points: \(pointsLeftHandConverted)")
                        print("Right hand points: \(pointsRightHandConverted)")
                        
                        //self.cameraViewClass.showPoints(pointsLeftHandConverted)
                        //self.cameraViewClass.showPoints(pointsRightHandConverted)
                        
                        if self.totalFrames.count < 4 {
                            let leftHandArray = self.convertToFloat32Array(points: pointsLeftHandConverted)
                            let rightHandArray = self.convertToFloat32Array(points: pointsRightHandConverted)
                            
                            self.totalFrames.append(leftHandArray + rightHandArray)
                        } else {
                            self.commentsContainerView.text = self.videoAttempt(videoFrames: self.totalFrames)
                            self.totalFrames = []
                        }
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
    
    // MARK: - Translation Functions

    // Function to read JSON files from the main bundle
    func loadJSON<T: Decodable>(filename: String, as type: T.Type) -> T? {
        guard let url = Bundle.main.url(forResource: filename, withExtension: "json") else {
            print("Failed to locate \(filename) in bundle.")
            return nil
        }
        
        do {
            let data = try Data(contentsOf: url)
            let decodedData = try JSONDecoder().decode(T.self, from: data)
            return decodedData
        } catch {
            print("Failed to decode \(filename): \(error)")
            return nil
        }
    }

    // Modify the videoAttempt function
    func videoAttempt(videoFrames: [[Float32]]) -> String? {
        // Load JSON data
        guard let inferenceArgs: InferenceArgs = loadJSON(filename: "inference_args", as: InferenceArgs.self),
              let characterMap: CharacterMap = loadJSON(filename: "character_to_prediction_index", as: CharacterMap.self) else {
            return nil
        }

        let selectedColumns = inferenceArgs.selected_columns
        let characterMapping = characterMap.map

        var dataFrame: [[Float32]] = []
        
        let faceArr = Array(repeating: Float32(0.0), count: 468)
        let poseArr = Array(repeating: Float32(0.0), count: 33)
        let handArr = Array(repeating: Float32(0.0), count: 21)

        // Process each frame in the video
        for frame in videoFrames {
            var row: [Float32] = []
            
            row.append(contentsOf: faceArr)
            row.append(contentsOf: frame[0..<21])
            row.append(contentsOf: poseArr)
            row.append(contentsOf: frame[21..<42])
            row.append(contentsOf: faceArr)
            row.append(contentsOf: frame[0..<21])
            row.append(contentsOf: poseArr)
            row.append(contentsOf: frame[21..<42])
            row.append(contentsOf: faceArr)
            row.append(contentsOf: handArr)
            row.append(contentsOf: poseArr)
            row.append(contentsOf: handArr)
            
            dataFrame.append(row)
        }
        
        guard let firstFrame = dataFrame.first, firstFrame.count == 1629 else {
            print("Error: No frame with the expected size of 1629.")
            return nil
        }
        
        let finalData: [Float32]
        finalData = firstFrame

        // Load the TFLite model
        guard let modelPath = Bundle.main.path(forResource: "model", ofType: "tflite") else {
            print("Failed to load the model file.")
            return nil
        }
        
        var options = Interpreter.Options()
        options.threadCount = 4
        let interpreter: Interpreter
        do {
            interpreter = try Interpreter(modelPath: modelPath, options: options)
            try interpreter.allocateTensors()
            
            print(try interpreter.input(at: 0).shape)
        } catch {
            print("Error creating TensorFlow Lite interpreter: \(error)")
            return nil
        }
        
        // Prepare the input tensor
        let flattenedData = finalData.compactMap { $0 }
        let inputData = Data(copyingBufferOf: flattenedData)

        do {
            try interpreter.copy(inputData, toInputAt: 0)
            try interpreter.invoke()
            
            let outputTensor = try interpreter.output(at: 0)
            let outputArray = [Float32](unsafeData: outputTensor.data) ?? []
            
            // Map the output to characters
            let predictionIndices = outputArray.map { Int($0) }
            
            // Create a reversed mapping from index to character
            let reversedCharacterMapping = Dictionary(uniqueKeysWithValues: characterMapping.map { ($0.value, $0.key) })

            // Convert prediction indices to characters using the reversed mapping
            let predictionString = predictionIndices.compactMap { reversedCharacterMapping[$0] }.joined()

            
            return predictionString
        } catch {
            print("Error running the model: \(error)")
            return nil
        }
    }
    
    // Define the structures for the JSON files
    struct InferenceArgs: Codable {
        let selected_columns: [String]
    }

    struct CharacterMap: Codable {
        let map: [String: Int]
    }
    
    func convertToFloat32Array(points: [CGPoint]) -> [Float32] {
        return points.flatMap { [Float32($0.x), Float32($0.y)] }
    }
}

// MARK: - Extensions

extension Data {
    init<T>(copyingBufferOf array: [T]) {
        self = array.withUnsafeBufferPointer { buffer in
            Data(buffer: buffer)
        }
    }
}

// Extension to convert Data to array
extension Array where Element: Numeric {
    init?(unsafeData: Data) {
        let elementSize = MemoryLayout<Element>.stride
        let count = unsafeData.count / elementSize
        guard unsafeData.count == count * elementSize else { return nil }
        
        self = unsafeData.withUnsafeBytes { buffer in
            Array<Element>(unsafeUninitializedCapacity: count) { pointer, initializedCount in
                buffer.copyBytes(to: pointer)
                initializedCount = count
            }
        }
    }
}
