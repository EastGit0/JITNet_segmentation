# Import socket module 
import socket             
  
# Create a socket object 
s = socket.socket()         
  
# Define the port on which you want to connect 
port = 12345                
  
# connect to the server on local computer 
s.connect(('35.233.229.168', port)) 

while True:
	# receive data from the server 
	print (s.recv(1024).decode())
	
# close the connection 
s.close()     