import os
import sys
import json

class Node:
    def __init__(self, id, value, parent=None):
        self.value = value
        self.id = id
        self.parent = parent

    def find_root(self):
        node = self
        while node.parent != None:
            node = node.parent
        return node    

    def __repr__(self,):
        return "id={},value={}".format(self.id, self.value)


class OntologyReader:
    def __init__(self, file):
        self.file = file
        lst_entries = json.loads(open(self.file).read())
        print("{} entries".format(len(lst_entries)))
        self.nodes = {}
        for entry in lst_entries:
            self.nodes[entry["id"]] = Node(id=entry["id"], value=entry["name"])
        for entry in lst_entries:
            parent_node = self.nodes[entry["id"]]
            for child_id in entry["child_ids"]:
                child_node = self.nodes[child_id]
                child_node.parent = parent_node
        print(self.nodes["/m/05zppz"].find_root())                
        


if __name__ == "__main__":
    ontology = OntologyReader("ontology.json")
    input_file, out_file = sys.argv[1], sys.argv[2]
    duration = 0
    with open(input_file) as f:
        with open(out_file, "w") as out:
            for l in f:
                l = l.strip()
                if l[0].startswith("#"):
                    continue
                t = l.split(', ')
                classes = t[-1].replace('\"',"").split(',')
                if not any([ontology.nodes[c].find_root().id==c for c in classes]):
                    out.write(l + '\n')
                    duration += (float(t[2]) - float(t[1]))
    print("Duration: {:.2f}".format(duration/60/60))
