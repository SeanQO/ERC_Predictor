from flask_table import Table, Col

class ItemTable(Table):
    name = Col('Cl')
    description = Col('Description')

# Get some objects
class Item(object):
    def __init__(self, name, description):
        self.name = name
        self.description = description

def get_table():
    items = [Item('Name1', 'Description1'),
         Item('Name2', 'Description2'),
         Item('Name3', 'Description3')]
    # Or, equivalently, some dicts
    items = [dict(name='Name1', description='Description1'),
         dict(name='Name2', description='Description2'),
         dict(name='Name3', description='Description3')]
    # Populate the table
    table = ItemTable(items)

    return table.__html__()

##https://flask-table.readthedocs.io/en/stable/
