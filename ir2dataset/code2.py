import csv

f=open('movies.csv','r')
k=csv.reader(f)

for i in k:
    print(i)
