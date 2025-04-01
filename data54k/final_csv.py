import pandas as pd
import csv

# Load all CSVs
people_df = pd.read_csv("combined_people.csv", on_bad_lines='skip')
uniqe_job_df=pd.read_csv("uniqe_titles.cvs")
job_list=uniqe_job_df["title"].dropna().astype(str).tolist()
clustered_job_titles_df=pd.read_csv("clustered_technical_job_titles_csv.csv")
#print(clustered_job_titles_df.head())

final_data=[]

for index, user in people_df.iterrows():
    skip=0
    new_user=[]
    id=user["person_id"]
    user_education=user["education"]
    user_experience=user["experience"].split(", ")
    user_skills=user["skills"]


    user_final_experience=[]
    #experience substring
    for title in user_experience:
        for job in job_list:
            #checking if job is a substring of title
            if title==job or job in title:
                    user_final_experience.append(job)
                    break
    
    user_clustered_jobs=[]
    #user clusters
    for title in user_final_experience:
        filtered = clustered_job_titles_df[clustered_job_titles_df["title"] == title]
        if not filtered.empty:
            job_row = filtered.iloc[0]
            cluster=job_row["United title"]
            user_clustered_jobs.append(cluster)
        else:
            skip=1

    if skip==0:
        string_exp=",".join([str(s) for s in user_final_experience])
        str_cluster=",".join([str(s) for s in user_clustered_jobs])
        new_user=[id,user_education,string_exp ,str_cluster, user_skills]
        final_data.append(new_user)

headers = ["person_id","Education", "Experience", "Clustered Experience", "Skills"]

# Writing to a CSV file
with open('final_users.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # Write the header row
    writer.writerows(final_data)  


        
     


