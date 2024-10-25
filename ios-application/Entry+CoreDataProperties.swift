//
//  Entry+CoreDataProperties.swift
//  FingerSpelling
//
//  Created by Shreshth Saharan on 23/4/2024.
//
//

import Foundation
import CoreData


extension Entry {

    @nonobjc public class func fetchRequest() -> NSFetchRequest<Entry> {
        return NSFetchRequest<Entry>(entityName: "Entry")
    }

    @NSManaged public var content: String?

}

extension Entry : Identifiable {

}
