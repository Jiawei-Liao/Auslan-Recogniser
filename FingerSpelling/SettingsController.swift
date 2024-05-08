//
//  SettingsController.swift
//  FingerSpelling
//
//  Created by Shreshth Saharan on 23/4/2024.
//

import UIKit

class SettingsController: UIViewController {
    
    @IBOutlet weak var rightHandSwitch: UISwitch!
    @IBOutlet weak var frontCameraSwitch: UISwitch!
    @IBOutlet weak var detectionModeSegment: UISegmentedControl!
    
    weak var databaseController: DatabaseProtocol?
    
    @IBAction func setRightHand(_ sender: Any) {
        databaseController?.updateSettings(mode: Int32(detectionModeSegment.selectedSegmentIndex), rightHand: (sender as AnyObject).isOn, frontCamera: frontCameraSwitch.isOn)
    }
    
    @IBAction func setFrontCamera(_ sender: Any) {
        databaseController?.updateSettings(mode: Int32(detectionModeSegment.selectedSegmentIndex), rightHand: rightHandSwitch.isOn, frontCamera: (sender as AnyObject).isOn)
    }
    
    @IBAction func setDetectionMode(_ sender: Any) {
        databaseController?.updateSettings(mode: Int32((sender as AnyObject).selectedSegmentIndex), rightHand: rightHandSwitch.isOn, frontCamera: frontCameraSwitch.isOn)
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let appDelegate = UIApplication.shared.delegate as? AppDelegate
        databaseController = appDelegate?.databaseController
        
        if let settings = databaseController?.fetchSettings() {
            rightHandSwitch.isOn = settings.rightHand
            frontCameraSwitch.isOn = settings.frontCamera
            detectionModeSegment.selectedSegmentIndex = Int(settings.mode)
        }
    }
}
