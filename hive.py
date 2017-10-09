from pyhive import presto

cursor = presto.connect('hiveserver-sap.s3s.altiscale.com').cursor()
cursor.execute('SELECT * FROM TABLE')
print(cursor.fetchone())
print(cursor.fetchall())
