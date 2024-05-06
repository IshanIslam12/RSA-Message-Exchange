from ai265 import *

#insert sender's public key in "person"
person = RsaPublicKey.read_from("""

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
