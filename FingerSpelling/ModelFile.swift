////
////  ModelFile.swift
////  FingerSpelling
////
////  Created by Shreshth Saharan on 6/9/2024.
////
//
//import Foundation
//import Firebase
//
//guard let modelPath = Bundle.main.path(forResource: "model", ofType: "tflite", inDirectory: "Model") 
//else { /* Handle error. */ }
//let localModel = CustomLocalModel(modelPath: modelPath)
//
//let interpreter = ModelInterpreter.modelInterpreter(localModel: localModel)
//
//NotificationCenter.default.addObserver(
//    forName: .firebaseMLModelDownloadDidSucceed,
//    object: nil,
//    queue: nil
//) { [weak self] notification in
//    guard let strongSelf = self,
//        let userInfo = notification.userInfo,
//        let model = userInfo[ModelDownloadUserInfoKey.remoteModel.rawValue]
//            as? RemoteModel,
//        model.name == "your_remote_model"
//        else { return }
//    // The model was downloaded and is available on the device
//}
//
//NotificationCenter.default.addObserver(
//    forName: .firebaseMLModelDownloadDidFail,
//    object: nil,
//    queue: nil
//) { [weak self] notification in
//    guard let strongSelf = self,
//        let userInfo = notification.userInfo,
//        let model = userInfo[ModelDownloadUserInfoKey.remoteModel.rawValue]
//            as? RemoteModel
//        else { return }
//    let error = userInfo[ModelDownloadUserInfoKey.error.rawValue]
//    // ...
//}

//        let ioOptions = ModelInputOutputOptions()
//        do {
//            try ioOptions.setInputFormat(index: 0, type: Float32, dimensions: [1, 224, 224, 3])
//            try ioOptions.setOutputFormat(index: 0, type: Float32, dimensions: [1, 1000])
//        } catch let error as NSError {
//            print("Failed to set input or output format with error: \(error.localizedDescription)")
//        }
