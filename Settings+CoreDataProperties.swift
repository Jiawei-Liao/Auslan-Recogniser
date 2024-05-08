//
//  Settings+CoreDataProperties.swift
//  FingerSpelling
//
//  Created by Shreshth Saharan on 23/4/2024.
//
//

import Foundation
import CoreData


extension Settings {

    @nonobjc public class func fetchRequest() -> NSFetchRequest<Settings> {
        return NSFetchRequest<Settings>(entityName: "Settings")
    }

    @NSManaged public var rightHand: Bool
    @NSManaged public var mode: Int32
    @NSManaged public var frontCamera: Bool

}

extension Settings : Identifiable {

}

enum Mode: Int32 {
    case single = 0
    case multi = 1
    case autocorrect = 2
}

extension Settings {
    var detectionmode: Mode {
        get {
            return Mode(rawValue: self.mode)!
        }
        
        set {
            self.mode = newValue.rawValue
        }
    }
}
