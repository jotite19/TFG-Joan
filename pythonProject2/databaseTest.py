from ase.db import connect

# Connexió a la base de dades QM9
db = connect('qm9.db')

row = db.get(id=1)
print(row)
for key in row:
    print('{0:30}: {1}'.format(key, row[key]))

print()
data = row.data
for item in data:
    print('{0:30}: {1}'.format(item, data[item]))
