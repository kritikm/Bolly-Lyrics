import os
import re
temp = open("fwithb.txt","w+")
flag=0
for filename in os.listdir(os.getcwd()):
	file1=open(filename, 'r')
	for line in file1.readlines():
		t=bool(re.search(r'\d',line))
#	for i in line:
#			if i=='(':
		if t is True:
			temp.write(filename+'\n')
			flag=1
			break
		if flag==1:
			break
	flag=0
				
