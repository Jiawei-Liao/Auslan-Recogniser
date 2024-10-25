//
//  CoreDataController.swift
//  FIT3178-W03-Lab
//
//  Created by Shreshth Saharan on 23/3/2024.
//

import UIKit
import CoreData

class CoreDataController: NSObject, DatabaseProtocol, NSFetchedResultsControllerDelegate {
    
    var entriesFetchedResultsController: NSFetchedResultsController<Entry>?
    
    var listeners = MulticastDelegate<DatabaseListener>()
    var persistentContainer: NSPersistentContainer
    
    override init() {
        persistentContainer = NSPersistentContainer(name: "FingerSpelling-DataModel")
        persistentContainer.loadPersistentStores() { (description, error ) in
            if let error = error {
                fatalError("Failed to load Core Data Stack with error: \(error)")
            }
        }
        
        super.init()
        
        if fetchAllEntries().count == 0 {
            createDefaultEntries()
        }
    }
    
    // MARK: - Settings methods
    
    func updateSettings(mode: Int32, rightHand: Bool, frontCamera: Bool) {
        let fetchRequest: NSFetchRequest<Settings> = Settings.fetchRequest()
            
        do {
            let settings = try persistentContainer.viewContext.fetch(fetchRequest)
                
            if let setting = settings.first {
                setting.mode = mode
                setting.rightHand = rightHand
                setting.frontCamera = frontCamera
            } else {
                let newSetting = Settings(context: persistentContainer.viewContext)
                newSetting.mode = mode
                newSetting.rightHand = rightHand
                newSetting.frontCamera = frontCamera
            }
                
            cleanup()
                
        } catch {
            print("Failed to fetch settings: \(error)")
        }
    }
    
    func fetchSettings() -> Settings? {
        let fetchRequest: NSFetchRequest<Settings> = Settings.fetchRequest()
        
        do {
            let settings = try persistentContainer.viewContext.fetch(fetchRequest)
            return settings.first
        } catch {
            print("Failed to fetch settings: \(error)")
            return nil
        }
    }
    
    // MARK: - Database methods
    
    func cleanup() {
        if persistentContainer.viewContext.hasChanges {
            do {
                try persistentContainer.viewContext.save()
            } catch {
                fatalError("Failed to save changes to Core Data with error: \(error)")
            }
        }
    }
    
    func addListener(listener: DatabaseListener) {
        listeners.addDelegate(listener)
        
        if listener.listenerType == .entry || listener.listenerType == .all { 
            listener.onEntriesChange(change: .update, entries: fetchAllEntries())
        }
    }
    
    func removeListener(listener: any DatabaseListener) {
        listeners.removeDelegate(listener)
    }
    
    // MARK: - Entry methods
    
    func addEntry(content: String) -> Entry {
        let entry = NSEntityDescription.insertNewObject(forEntityName: "Entry", into: persistentContainer.viewContext) as! Entry
        entry.content = content
        
        return entry
    }
    
    func deleteEntry(entry: Entry) {
        persistentContainer.viewContext.delete(entry)
    }
    
    func fetchAllEntries() -> [Entry] {
        if entriesFetchedResultsController == nil {
            let request: NSFetchRequest<Entry> = Entry.fetchRequest()
            let nameSortDescriptor = NSSortDescriptor(key: "content", ascending: true)
            request.sortDescriptors = [nameSortDescriptor]
            
            // Initialise Fetched Results Controller
            entriesFetchedResultsController = NSFetchedResultsController<Entry>(fetchRequest: request, managedObjectContext: persistentContainer.viewContext, sectionNameKeyPath: nil, cacheName: nil)
            
            // Set this class to be the results delegate
            entriesFetchedResultsController?.delegate = self
            
            do {
                try entriesFetchedResultsController?.performFetch()
            } catch {
                print("Fetch Request Failed: \(error)")
            }
        }
        
        if let entries = entriesFetchedResultsController?.fetchedObjects {
            return entries
        }
        
        return [Entry]()
    }
    
    // MARK: - Fetched Results Controller Protocol methods
    
    func controllerDidChangeContent(_ controller: NSFetchedResultsController<NSFetchRequestResult>) {
        if controller == entriesFetchedResultsController {
            listeners.invoke() { listener in
                if listener.listenerType == .entry || listener.listenerType == .all {
                    listener.onEntriesChange(change: .update, entries: fetchAllEntries())
                }
            }
        }
    }
    
    func createDefaultEntries() {
        let _ = addEntry(content: "Entry number one")
        let _ = addEntry(content: "Entry number two")
        let _ = addEntry(content: "Entry number three")
            
        cleanup()
    }
}
