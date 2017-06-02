import os
temp = open("del.txt", "w+")
temp.close()
flag=0
for filename in os.listdir(os.getcwd()):
	file = open(filename, 'r')
	cnt=1;
	for line in file.readlines():
		if cnt==1:
			line1=line
		if cnt==2:
			line2=line			
		cnt +=1
	line2=line2.rstrip()

	x=line2+';'+line1
	f=open("output.txt","r")
	for w in f.readlines():
		if w==x:
			g=open("del.txt","a+")
			g.write(filename)
			g.write("\n")
			g.close()
			flag=1
	f.close()
	if flag==0:	
		file2 = open("output.txt", "a+")
		file2.write(x)
		file2.close()
	file.close()
	flag=0
