//
//  DatabaseProtocol.swift
//  FIT3178-W03-Lab
//
//  Created by Shreshth Saharan on 23/3/2024.
//

import Foundation

enum DatabaseChange {
    case add
    case remove
    case update
}

enum ListenerType {
    case entry
    case all
}

protocol DatabaseListener: AnyObject {
    var listenerType: ListenerType {get set}
    func onEntriesChange(change: DatabaseChange, entries: [Entry])
}

protocol DatabaseProtocol: AnyObject {
    func cleanup()
    
    func addListener(listener: DatabaseListener)
    func removeListener(listener: DatabaseListener)
    
    func addEntry(content: String) -> Entry
    func deleteEntry(entry: Entry)
    
    func updateSettings(mode: Int32, rightHand: Bool, frontCamera: Bool)
    func fetchSettings() -> Settings?
}
