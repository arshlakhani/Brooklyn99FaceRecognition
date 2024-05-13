# faker - fake names
from faker import Faker
fake = Faker()
names=[fake.name().split() for i in range(100)]
names = [name for name in names if len(name)==2]
print(names)

import sqlite3
connection = sqlite3.connect("sample.db")
insert_query = 'INSERT INTO people(name, surname) VALUES(?,?)'
cursor = connection.cursor()
for name in names:
    cursor.execute(insert_query,name)
connection.commit()