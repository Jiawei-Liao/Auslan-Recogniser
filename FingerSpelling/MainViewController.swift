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
import SwiftCSV

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
    let maxFrames = 160
    
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
    func extractPoints(from observation: VNHumanHandPoseObservation, bodyObservation: VNHumanBodyPoseObservation?, forHand hand: inout [CGPoint], forBody body: inout [CGPoint], confidenceThreshold: Float) -> Bool {
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
                thumbCmcPoint, thumbMpPoint, thumbIpPoint, thumbTipPoint,
                indexMcpPoint, indexPipPoint, indexDipPoint, indexTipPoint,
                middleMcpPoint, middlePipPoint, middleDipPoint, middleTipPoint,
                ringMcpPoint, ringPipPoint, ringDipPoint, ringTipPoint,
                littleMcpPoint, littlePipPoint, littleDipPoint, littleTipPoint
            ]
            
            var i = 1
            
            for point in pointKeys {
                if point.confidence > confidenceThreshold {
                    hand[i] = CGPoint(x: point.location.x, y: point.location.y)
                    i += 1
                } else {
                    return false
                }
            }
            
            if let bodyObservation = bodyObservation {
                let bodyPoints = try bodyObservation.recognizedPoints(forGroupKey: VNHumanBodyPoseObservation.JointsGroupName.all.rawValue)
                
                guard let leftEye = bodyPoints[VNRecognizedPointKey(rawValue: VNHumanBodyPoseObservation.JointName.leftEye.rawValue.rawValue)],
                      let rightEye = bodyPoints[VNRecognizedPointKey(rawValue: VNHumanBodyPoseObservation.JointName.rightEye.rawValue.rawValue)],
                      let leftEar = bodyPoints[VNRecognizedPointKey(rawValue: VNHumanBodyPoseObservation.JointName.leftEar.rawValue.rawValue)],
                      let rightEar = bodyPoints[VNRecognizedPointKey(rawValue: VNHumanBodyPoseObservation.JointName.rightEar.rawValue.rawValue)],
                      let nose = bodyPoints[VNRecognizedPointKey(rawValue: VNHumanBodyPoseObservation.JointName.nose.rawValue.rawValue)],
                      let neck = bodyPoints[VNRecognizedPointKey(rawValue: VNHumanBodyPoseObservation.JointName.neck.rawValue.rawValue)],
                      let leftShoulder = bodyPoints[VNRecognizedPointKey(rawValue: VNHumanBodyPoseObservation.JointName.leftShoulder.rawValue.rawValue)],
                      let rightShoulder = bodyPoints[VNRecognizedPointKey(rawValue: VNHumanBodyPoseObservation.JointName.rightShoulder.rawValue.rawValue)],
                      let leftElbow = bodyPoints[VNRecognizedPointKey(rawValue: VNHumanBodyPoseObservation.JointName.leftElbow.rawValue.rawValue)],
                      let rightElbow = bodyPoints[VNRecognizedPointKey(rawValue: VNHumanBodyPoseObservation.JointName.rightElbow.rawValue.rawValue)],
                      let leftHip = bodyPoints[VNRecognizedPointKey(rawValue: VNHumanBodyPoseObservation.JointName.leftHip.rawValue.rawValue)],
                      let rightHip = bodyPoints[VNRecognizedPointKey(rawValue: VNHumanBodyPoseObservation.JointName.rightHip.rawValue.rawValue)],
                      let root = bodyPoints[VNRecognizedPointKey(rawValue: VNHumanBodyPoseObservation.JointName.root.rawValue.rawValue)],
                      let leftKnee = bodyPoints[VNRecognizedPointKey(rawValue: VNHumanBodyPoseObservation.JointName.leftKnee.rawValue.rawValue)],
                      let rightKnee = bodyPoints[VNRecognizedPointKey(rawValue: VNHumanBodyPoseObservation.JointName.rightKnee.rawValue.rawValue)],
                      let leftAnkle = bodyPoints[VNRecognizedPointKey(rawValue: VNHumanBodyPoseObservation.JointName.leftAnkle.rawValue.rawValue)],
                      let rightAnkle = bodyPoints[VNRecognizedPointKey(rawValue: VNHumanBodyPoseObservation.JointName.rightAnkle.rawValue.rawValue)]
                else {
                    print("Failed body point extraction")
                    return false
                }
                
                let bodyPointKeys = [
                    nose, leftEye, leftEye, leftEye, rightEye, rightEye,
                    rightEye, leftEar, rightEar, neck, neck,
                    leftShoulder, rightShoulder, leftElbow, rightElbow,
                    neck, neck, neck, neck, neck, neck, neck, neck,
                    leftHip, rightHip, leftKnee, rightKnee, leftAnkle, rightAnkle,
                    leftAnkle, rightAnkle, leftAnkle, rightAnkle
                ]
                
                for (i, point) in bodyPointKeys.enumerated() {
                    if point.confidence >= 0 /*confidenceThreshold*/ {
                        body[i] = CGPoint(x: point.location.x, y: point.location.y)
                    } else {
                        return false
                    }
                }
            }

            if wristPoint.confidence > confidenceThreshold {
                hand[0] = CGPoint(x: wristPoint.location.x, y: wristPoint.location.y)
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
        
        let bodyPoseRequest = VNDetectHumanBodyPoseRequest()
        
        do {
            try handler.perform([handPoseRequest, bodyPoseRequest])
            
            guard let handObservations = handPoseRequest.results, handObservations.count > 0,
                  let bodyObservations = bodyPoseRequest.results else {
                DispatchQueue.main.async {
                    //self.cameraViewClass.showPoints([])
                }
                return
            }

            var handOne: [CGPoint] = [CGPoint](repeating: CGPoint(), count: 21)
            var handTwo: [CGPoint] = [CGPoint](repeating: CGPoint(), count: 21)
            var bodyPoints: [CGPoint] = [CGPoint](repeating: CGPoint(), count: 33)

            let confidenceThreshold: Float = 0.2

            let handOneSuccess = extractPoints(from: handObservations[0], bodyObservation: bodyObservations.first, forHand: &handOne, forBody: &bodyPoints, confidenceThreshold: confidenceThreshold)
            var handTwoSuccess = false

            if handObservations.count > 1 {
                handTwoSuccess = extractPoints(from: handObservations[1], bodyObservation: nil, forHand: &handTwo, forBody: &bodyPoints, confidenceThreshold: confidenceThreshold)
            }

            if !handOneSuccess || (handObservations.count > 1 && !handTwoSuccess) {
                DispatchQueue.main.async {
                    //self.cameraViewClass.showPoints([])
                }
                print("Failed point extraction or confidence test")
                return
            }

            if handObservations.count == 2 {
                if true {
                    let handOneIsLeftHand = handOne[0].x < handTwo[0].x

                    // Bingo. The jumble happens here. handOne.values will return the unordered values.
                    let pointsLeftHand = handOneIsLeftHand ? handOne : handTwo
                    let pointsRightHand = handOneIsLeftHand ? handTwo : handOne
                    
                    // Adjustments as needed by the model
                    bodyPoints[15] = pointsLeftHand[0]
                    bodyPoints[16] = pointsRightHand[0]
                    bodyPoints[17] = pointsLeftHand[16]
                    bodyPoints[18] = pointsRightHand[16]
                    bodyPoints[19] = pointsLeftHand[7]
                    bodyPoints[20] = pointsRightHand[7]
                    bodyPoints[21] = pointsLeftHand[3]
                    bodyPoints[22] = pointsRightHand[3]

                    DispatchQueue.main.async {
//                        let pointsLeftHandConverted = pointsLeftHand.map {
//                            self.cameraViewClass.previewLayer.layerPointConverted(fromCaptureDevicePoint: $0)
//                        }
//
//                        let pointsRightHandConverted = pointsRightHand.map {
//                            self.cameraViewClass.previewLayer.layerPointConverted(fromCaptureDevicePoint: $0)
//                        }
                        
                        print("Left hand points: \(pointsLeftHand)")
                        print("Right hand points: \(pointsRightHand)")
                        print("Body points: \(bodyPoints)")
                        
//                        self.cameraViewClass.showPoints(pointsLeftHandConverted)
//                        self.cameraViewClass.showPoints(pointsRightHandConverted)
                        
                        if self.totalFrames.count < self.maxFrames {
                            let leftHandArray = self.convertToFloat32Array(points: Array(pointsLeftHand))
                            let rightHandArray = self.convertToFloat32Array(points: Array(pointsRightHand))
                            let bodyArray = self.convertToFloat32Array(points: Array(bodyPoints))
                            
                            self.totalFrames.append(leftHandArray + rightHandArray + bodyArray)
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
        
        let faceArr = Array(repeating: Float32.nan, count: 468)
        let poseArr = Array(repeating: Float32.nan, count: 33)
        let handArr = Array(repeating: Float32.nan, count: 21)
        
        func extractOddElements(from array: ArraySlice<Float32>) -> [Float32] {
            return array.enumerated().compactMap { (index, element) in
                return index % 2 == 1 ? element : nil
            }
        }

        func extractEvenElements(from array: ArraySlice<Float32>) -> [Float32] {
            return array.enumerated().compactMap { (index, element) in
                return index % 2 == 0 ? element : nil
            }
        }

        // Process each frame in the video
        for frame in videoFrames {
            var row: [Float32] = []

            row.append(contentsOf: faceArr)
            row.append(contentsOf: extractOddElements(from: frame[0..<42]))  // only odd indices from 0 to 41
            row.append(contentsOf: extractOddElements(from: frame[85..<118]))  // only odd indices from 85 to 117
            row.append(contentsOf: extractEvenElements(from: frame[0..<42])) // only even indices from 0 to 41
            row.append(contentsOf: faceArr)
            row.append(contentsOf: extractOddElements(from: frame[42..<84])) // only odd indices from 42 to 83
            row.append(contentsOf: extractEvenElements(from: frame[85..<118])) // only even indices from 85 to 117
            row.append(contentsOf: extractEvenElements(from: frame[42..<84])) // only even indices from 42 to 83
            row.append(contentsOf: faceArr)
            row.append(contentsOf: handArr)
            row.append(contentsOf: poseArr)
            row.append(contentsOf: handArr)
            
            dataFrame.append(row)
        }

        // Ensure that the size of each frame is correct
        guard let firstFrame = dataFrame.first, firstFrame.count == 1629 else {
            print("Error: No frame with the expected size of 1629.")
            return nil
        }

        // Flatten all frames into a 1D array for TensorFlow Lite input
        let flattenedData = dataFrame.flatMap { $0 }

        // Load the TensorFlow Lite model
        guard let modelPath = Bundle.main.path(forResource: "model", ofType: "tflite") else {
            print("Failed to load the model file.")
            return nil
        }
        
        var options = Interpreter.Options()
        options.threadCount = 1
        let interpreter: Interpreter


        var inputData = Data(copyingBufferOf: flattenedData)
        
        // MARK: Uncomment this to run with the csv data.
        
//        do {
//            // Ensure the URL is unwrapped correctly
//            if let myUrl = Bundle.main.url(forResource: "out", withExtension: "csv") {
//                // Instantiate the CSV
//                let csvFile = try CSV<Named>(url: myUrl)
//
//                // Get the headers in order from the CSV
//                let headers = csvFile.header
//
//                // Convert rows to arrays of Floats while respecting the header order
//                let inputArray = csvFile.rows.map { dict in
//                    headers.map { header in
//                        Float(dict[header] ?? "") ?? Float.nan
//                    }
//                }
//
//                // Flatten the array of arrays into a single array and create inputData
//                inputData = Data(copyingBufferOf: inputArray.flatMap { $0 })
//
//            } else {
//                print("CSV file not found.")
//            }
//
//        } catch {
//            // Catch and handle errors from parsing invalid CSV
//            print("Error reading CSV: \(error)")
//        }
        
        do {
            interpreter = try Interpreter(modelPath: modelPath)
            
            print(interpreter.inputTensorCount)
            print(interpreter.outputTensorCount)
            
            try interpreter.resizeInput(at: 0, to: Tensor.Shape(1, inputData.count/4)) // float is 4 bytes, meaning that inputData is 4x too long.
        
            try interpreter.allocateTensors()

            // Copy input data into the interpreter
            try interpreter.copy(inputData, toInputAt: 0)
            
            // Run inference
            try interpreter.invoke()
            
            // Get the output tensor
            let outputTensor = try interpreter.output(at: 0)
            
            // Convert the output data to an array
            let outputArray = [Float32](unsafeData: outputTensor.data) ?? []
            
            // Map the output to characters
            let predictionIndices = outputArray.map { Int($0) }
            
            // Create a reversed mapping from index to character
            let reversedCharacterMapping = Dictionary(uniqueKeysWithValues: characterMapping.map { ($0.value, $0.key) })

            var resultString = ""

            // Iterate through the output array in chunks of 59
            let chunkSize = 59
            for chunkStart in stride(from: 0, to: outputArray.count, by: chunkSize) {
                // Get the current chunk (subarray of 59 values)
                let chunk = Array(outputArray[chunkStart..<min(chunkStart + chunkSize, outputArray.count)])
                
                // Find the index where the value is 1
                if let index = chunk.firstIndex(of: 1) {
                    // Look up the corresponding character from the reversed map
                    if let character = reversedCharacterMapping[index] {
                        // Append the character to the result string
                        resultString.append(character)
                    }
                }
            }

            // Print the final result string
            print("Resulting String:", resultString)
            
            return resultString
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
