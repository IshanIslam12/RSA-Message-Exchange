from ai265 import *

#insert sender's public key in "person"
person = RsaPublicKey.read_from("""
-----BEGIN RSAISH PUBLIC KEY-----
Bits: 2044
Owner: Chris League
Generated: 2024-04-28 23:22:21+00:00

AwABAAEAAUOuE/S0fEKARV7zJqLiAyneqcYixvyr3aUJvTF7
dbAw2Hy0RSbDyH8dBSNh8J55PJqiElFJmGWm4Ld45yBjbU8W
cdsvd7hXHw2UGX6BYhkNsNeOZa/ALnwM1KXhFP7D6p+oR8by
0aWHITJUhaXXUPZboa/5H+kBEkBwsTHvEdQmoPZ+57+Vbs+3
ge5RHLiicXszg8kIa7L6r/A5gipQ5cRHYNrZjLVuFgGJiGdL
Y1OcT/cGY6vsZhSR2jiiUHxknMC9n2WFNFFshk0w/X2rLOyE
Vmp/X4PXBlQV9yXq5jFklPutFNyYu7GwmYkQPzxLGcTjzVSi
ehb599gg8DP/rwk=
-----END RSAISH PUBLIC KEY-----

""")

#generation of public and private keys
# note "private_key" when you generate public and private keys
ishan = RsaKeyPair.generate(who="Ishan")
print(ishan) # You keep this result
print(ishan.pub) # You send/post this result


#insert private key in "private_key"(when generated and remeber to unnote when done)
# note "private_key" when you generate public and private keys
#private_key = RsaKeyPair.read_from("""

#""")

#encryption of message
enc = io.StringIO()
RSA.sign_encrypt_text(ishan, person, "This is my proof. I blew a bubble in the park", enc)
msg = enc.getvalue()
print(msg)



#insert encrypted message from other party in "msg2"
msg2 = """

"""

#decryption of message(unote when time to decrypt)
#RSA.decrypt_print_text(person, ishan, msg2)