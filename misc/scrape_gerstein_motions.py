import requests
import pandas as pd
from bs4 import BeautifulSoup
root_link = "http://www.molmovdb.org/cgi-bin"
r = requests.get(root_link + "/browse.cgi")
soup = BeautifulSoup(r.text, "html.parser")

motion_links = [link['href'] for link in soup.find_all('a', href=True) if "motion.cgi" in link['href']]
motion_classes = []
struct_1 = []
struct_2 = []
for link in motion_links:
    motion_r = requests.get(link)
    motion_soup = BeautifulSoup(motion_r.text, "html.parser")
    
    tables = motion_soup.find_all("table")
    # find first table with text Classification
    class_tables = [table for table in tables if "Classification" in table.get_text()]
    if(len(class_tables) < 1): 
        continue
    class_table = class_tables[0]
   
    # find all tables with text Structures
    structure_tables = [table for table in tables if "Structures" in table.get_text()]
    if(len(structure_tables) < 1): 
        continue
    structure_table = structure_tables[0]
    classes = [a.get_text() for a in class_table.find_all("a", string=True)]
    structures = [a.get_text() for a in structure_table.find_all("a", string=True)]
    if(len(classes) != 1 or len(structures) != 2):
        continue
    motion_classes.append(classes[0])
    struct_1.append(structures[0])
    struct_2.append(structures[1])

out_df = pd.DataFrame({"motion_class":motion_classes, "struct_1":struct_1, "struct_2":struct_2})
out_df.to_csv("gerstein.csv")
        
    
      

