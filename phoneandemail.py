#! /usr/bin/env python3
import re, pyperclip

#REGEX FOR phone
phone_regex = re.compile(r'''
# 415-555-0000, 555-0000, (415) 555-0000, 555-0000 ext 12345, ext. 12345, x12345
(
((\d\d\d)|(\(\d\d\d\)))? # area code (optional)
(\s|-)    # first seperator
\d\d\d    # first three digits
(\s|-)    # seperator
\d\d\d\d    # last 4 digits
(((ext(\.)?\s) |x)    # extension word part(optional)
(\d{2, 5}))? #extension number-part(also optional)
)
''', re.VERBOSE)
#REGEX FOR EMAIL
email_regex = re.compile(r'''
[a-zA-Z0-9_.+]+          #name part
@     #@ aymbol
[a-zA-Z0-9_.+]+  #domainname part
''', re.VERBOSE)

#GEt txt off clipboard
text = pyperclip.paste()
#extract txt
extracted_phone = phone_regex.findall(text)
extracted_email = email_regex.findall(text)

allPhoneNumbers = []
for phonenumber in extracted_phone:
    allPhoneNumbers.append(phonenumber[0])
                           

#print(extracted_email)
#print(allPhoneNumbers)
#copy extracted text to clipboard
results = '\n'.join(allPhoneNumbers)+'\n'+'\n'.join(extracted_email)
pyperclip.copy(results)
