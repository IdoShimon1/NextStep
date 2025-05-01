##changing all labels to fit their id 
import json
import pandas as pd
from collections import Counter

filename = "Data/Web Designer.json"
label_name = "Web Designer"  #filename.replace('.json', '')
# Load the JSON file

# Read the raw file content
with open(filename, 'r', encoding='utf-8') as f:
    raw = f.read()

# Parse each JSON object individually
data = []
decoder = json.JSONDecoder()
pos = 0
length = len(raw)

while pos < length:
    try:
        obj, size = decoder.raw_decode(raw[pos:])
        for obj in data:  # data is the JSON list
           obj['label'] = label_name

        data.append(obj)
        pos += size

        # Skip whitespace or newlines
        while pos < length and raw[pos].isspace():
            pos += 1
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON at position {pos}: {e}")
        break

# Example simplified mapping (use full version in real case)
degree_group_map = {
    #high school
    "high school": "High School",
    "high school diploma": "High School",
    "a level": "High School",
    "a-level": "High School",
    "a+": "High School",
    "secondary school": "High School",
    "secondary special education": "High School",
    "leaving certificate": "High School",
    "higher secondary": "High School",
    "ict-beheerder niv. 4": "High School",
    "ict-beheerder, medewerker beheerder ict": "High School",
    "intermediate": "High School",
    "intermediate computer sciences": "High School",
    "nfq level 5 qqi award": "High School",

    #Associate Degrees
    "associate degree":"Associate Degrees",
    "associates of science":"Associate Degrees",
    "associate of arts":"Associate Degrees",
    "pursued associate degree":"Associate Degrees",
    "cas":"Associate Degrees",

    #Bachelor of Science 
    "bachelor of science":"Bachelor of Science",
    "b.sc.":"Bachelor of Science",
    "bsc":"Bachelor of Science",
    "bs":"Bachelor of Science",
    "b.s.":"Bachelor of Science",
    "b. sc.":"Bachelor of Science",
    "bachelor of science (bs/bsc/b.sc./honours/in progress)":"Bachelor of Science",
    "bachelor of sciences":"Bachelor of Science",
    "bachelor of applied science (b.a.sc.)":"Bachelor of Science",
    "bachelors of science":"Bachelor of Science",
    "honours bachelor of science (b.sc)":"Bachelor of Science",
    "bachelor of applied arts and sciences":"Bachelor of Science",
    "bachelor of science with honours":"Bachelor of Science",
    "bachelor of science - bs/bsc (hons)":"Bachelor of Science",
    "bachelor of computer science":"Bachelor of Science",
    "bachelor of cyber security":"Bachelor of Science",
    "bachelor of information systems":"Bachelor of Science",
    "bachelor of information technology":"Bachelor of Science",
    "bachelor of mathematics":"Bachelor of Science",

    #Bachelor of Arts
    "bachelor of arts":"Bachelor of Arts",
    "b.a":"Bachelor of Arts",
    "ba":"Bachelor of Arts",
    "ba (hons)":"Bachelor of Arts",
    "bachelor of arts - honours":"Bachelor of Arts",
    "bachelor of education - be":"Bachelor of Arts",
    "bachelor of surveying":"Bachelor of Arts",

    #Bachelor of Business
    "bachelor of business administration":"Bachelor of Business",
    "b.b.a.":"Bachelor of Business",
    "bba":"Bachelor of Business",
    "b.com.":"Bachelor of Business",
    "llb":"Bachelor of Business",
    "ll.b.":"Bachelor of Business",
    "bachelor of commerce":"Bachelor of Business",
    "bachelor of business information systems":"Bachelor of Business",
    "bachelors in business":"Bachelor of Business",
    "bachelor of laws":"Bachelor of Business",

    #Bachelor of Engineering
    "bachelor of engineering":"Bachelor of Engineering",
    "be":"Bachelor of Engineering",
    "b.e.":"Bachelor of Engineering",
    "beng":"Bachelor of Engineering",
    "bachelor of engineering - be/beng":"Bachelor of Engineering",
    "b.tech.":"Bachelor of Engineering",
    "btech":"Bachelor of Engineering",
    "b. tech":"Bachelor of Engineering",
    "b-tech":"Bachelor of Engineering",
    "bachelor of technology":"Bachelor of Engineering",
    "bachelor of technology & applied science":"Bachelor of Engineering",
    "btech - bachelor of technology":"Bachelor of Engineering",
    "engineer's degree":"Bachelor of Engineering",
    "software engineer's degree":"Bachelor of Engineering",
    "practical engineer":"Bachelor of Engineering",
    "software practical engineer":"Bachelor of Engineering",
    "practical engineering":"Bachelor of Engineering",

    #Master of Science
    "master of science":"Master of Science",
    "m.sc.":"Master of Science",
    "ms":"Master of Science",
    "m.s.":"Master of Science",
    "msc":"Master of Science",
    "msc/ms":"Master of Science",
    "master's of science":"Master of Science",
    "master of science in information technology":"Master of Science",
    "master of technology - mtech":"Master of Science",
    "master's degree":"Master of Science",
    "master’s of science":"Master of Science",
    "master’s degree":"Master of Science",
    "msc+mtech dual degree":"Master of Science",
    "master of computer applications":"Master of Science",
    "master's":"Master of Science",
    "masters":"Master of Science",

    #Master of Arts
    "master of arts":"Master of Arts",
    "ma":"Master of Arts",
    "m.a.":"Master of Arts",
    "ma hons":"Master of Arts",
    "master of divinity (m.div.)":"Master of Arts",

    #Business Master's
    "master of business administration":"Business Master's",
    "mba ":"Business Master's",
    "m.b.a.":"Business Master's",
    "master's of business administration":"Business Master's",
    "tm/mba":"Business Master's",
    "master's of management":"Business Master's",
    "master of management analytics":"Business Master's",
    "master of public administration":"Business Master's",
    "master of public policy":"Business Master's",
    "master of international business":"Business Master's",

    #Technology Master’s
    "master of engineering":"Technology Master's",
    "meng ":"Technology Master's",
    "m.eng.":"Technology Master's",
    "master of technology":"Technology Master's",
    "mtech":"Technology Master's",
    "master's of information technology":"Technology Master's",
    "master's of professional studies":"Technology Master's",
    "masters of technology management":"Technology Master's",

    #Doctorate
    "doctor of philosophy":"Doctorate",
    "ph.d.":"Doctorate",
    "phd":"Doctorate",
    "ph.d. and s.m.":"Doctorate",
    "phd researcher":"Doctorate",
    "doctorant":"Doctorate",

    #Certificate
    "diploma":"Certificate",
    "extended diploma":"Certificate",
    "graduate diploma":"Certificate",
    "graduate diploma of computing":"Certificate",
    "postgraduate degree":"Certificate",
    "post graduate diploma":"Certificate",
    "post-graduate diploma":"Certificate",
    "post graduation diploma":"Certificate",
    "graduate certificate":"Certificate",
    "graduate certificate - cyber security":"Certificate",
    "graduate certificate program":"Certificate",
    "postgraduate certificate in education":"Certificate",
    "foundation degree":"Certificate",
    "research diploma":"Certificate",
    "diploma of education":"Certificate",
    "diploma in pc engineering":"Certificate",
    "new zealand diploma in networking":"Certificate",
    "certificate":"Certificate",
    "certification / certifications":"Certificate",
    "cybersecurity certificate":"Certificate",
    "data science professional certificate":"Certificate",
    "executive certificate":"Certificate",
    "professional course":"Certificate",
    "refresher course":"Certificate",
    "certified hands-on cyber security specialist (chcss)":"Certificate",
    "ceh":"Certificate",
    "ccna":"Certificate",
    "ccna & cyberops":"Certificate",
    "comptia+":"Certificate",
    "crte":"Certificate",
    "istqb":"Certificate",
    "mcse":"Certificate",
    "network+":"Certificate",
    "lpi linux essentials":"Certificate",
    "sans for508":"Certificate",
    "tciso - technion certified information security manager":"Certificate",
    "micromasters program":"Certificate",
    "microsoft azure data fundamental":"Certificate",
    "python development":"Certificate",
    "practitioner":"Certificate",
    "course / courseware":"Certificate",
    "nanodegree":"Certificate",
    "bootcamp / boot camp / immersive bootcamp":"Certificate",
    "data bootcamp / cyber bootcamp / programming":"Certificate",

    #Cybersecurity 
    "cyber security":"Certificate",
    "cyber security architecture & design":"Certificate",
    "cyber security expert":"Certificate",
    "cyber security operations":"Certificate",
    "cyber/electronic operations and warfare":"Certificate",
    "it/cyber security":"Certificate",
    "cybersecurity immersive bootcamp":"Certificate",
    "cybersecurity professional bootcamp":"Certificate",
    "cybersecurity certificate":"Certificate",
    "information technology management/project management":"Certificate",
    "cyber management":"Certificate",
    "advanced penetration testing":"Certificate",
    "penetration tester/testing":"Certificate",
    "incident response":"Certificate",
    "qa engineer course / manual / tester":"Certificate",
    "ciso / ciso & dpo / tciso":"Certificate",
    "csp / csp: cyber security practitioner":"Certificate",

    #Dev
    "android dev":"Certificate",
    "advanced backend developer":"Certificate",
    "fullstack / fullstack developer / full-stack development":"Certificate",
    "frontend developer":"Certificate",
    "full stack web developer":"Certificate",
    "web development":"Certificate",
    "system engineer":"Certificate",
    "system and network practitioner":"Certificate",
    "responsible en ingénierie systèmes et réseaux":"Certificate",

    #null
    "music business":"null",
    "architectural design and technology":"null",
    "russian studies":"null",
    "chinese language studies":"null",
    "emergency medical technology/technician (emt paramedic)":"null",
    "self-educated":"null",
    "n/a":"null",
    "null":"null",
    "1":"null",
    "degree":"null"




}

# Normalize mapping keys
degree_group_map = {k.strip().lower(): v for k, v in degree_group_map.items()}


for user in data:
    if 'education' in user:
        for edu in user['education']:
            original_degree = edu.get('degree', '')
            if original_degree:
                original_degree=original_degree.strip().lower()
            group = degree_group_map.get(original_degree, 'Other / Unknown')
            edu['degree'] = group


# Save all cleaned objects as a valid JSON array
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✅ Processed {len(data)} JSON objects and added label: '{label_name}'")

