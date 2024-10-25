//
//  HistoryController.swift
//  FingerSpelling
//
//  Created by Shreshth Saharan on 23/4/2024.
//

import UIKit

class HistoryController: UITableViewController, DatabaseListener {
    
    var listenerType: ListenerType = .entry
    weak var databaseController: DatabaseProtocol?

    let CELL_ENTRY = "contentCell"
    
    var allEntries: [Entry] = []

    override func viewDidLoad() {
        super.viewDidLoad()
        
        let appDelegate = UIApplication.shared.delegate as? AppDelegate
        databaseController = appDelegate?.databaseController
    }

    // MARK: - Table view data source

    override func numberOfSections(in tableView: UITableView) -> Int {
        return 1
    }

    override func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return allEntries.count
    }

    override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: CELL_ENTRY, for: indexPath)
                    
        var content = cell.defaultContentConfiguration()
        let entry = allEntries[indexPath.row]
        content.text = entry.content
        cell.contentConfiguration = content
        
        return cell
    }

    override func tableView(_ tableView: UITableView, canEditRowAt indexPath: IndexPath) -> Bool {
        return true
    }

    override func tableView(_ tableView: UITableView, commit editingStyle: UITableViewCell.EditingStyle, forRowAt indexPath: IndexPath) {
        if editingStyle == .delete {
            let entry = allEntries[indexPath.row]
            databaseController?.deleteEntry(entry: entry)
        }
    }

    // MARK: - DatabaseListener methods
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        databaseController?.addListener(listener: self)
    }
        
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        databaseController?.removeListener(listener: self)
    }
    
    func onEntriesChange(change: DatabaseChange, entries: [Entry]) {
        allEntries = entries
        tableView.reloadData()
    }
}
